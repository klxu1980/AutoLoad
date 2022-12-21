import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


class CoilOffset():
    def __init__(self, frame_size, frame_cnt):
        mod = SourceModule("""
            __global__ void CalcDiffers(unsigned char *lst_frame, unsigned char *cur_frames, int *tmp, int *RMSE, int *input_size)
            {
                int frame_cnt  = input_size[0];
                int frame_size = input_size[1];
                int base = blockIdx.x * frame_size;
        
                // 计算当前帧与上一帧每个像素差值的平方
                int tid = threadIdx.x;
                while(tid < frame_size)
                {
                    int diff = cur_frames[base + tid] - lst_frame[tid];
                    tmp[base + tid] = diff * diff;
                    tid += blockDim.x;
                }
                __syncthreads();
                
                // 计算求和
                int offset = frame_size / 2;
                while(offset)
                {
                    tid = threadIdx.x;
                    while(tid < offset)
                    {
                        tmp[base + tid] += tmp[base + tid + offset];
                        tid += blockDim.x;
                    }
                    __syncthreads();
                    
                    offset /= 2;
                }
                
                // 计算结果输出到RMSE中
                if(threadIdx.x == 0)
                    RMSE[blockIdx.x] = tmp[blockIdx.x * frame_size];
            }
            """)
        self.func = mod.get_function("CalcDiffers")

        self.frame_size = frame_size
        self.frame_cnt = frame_cnt

        # 分配cuda全局内存
        self.lst_frame_gpu = cuda.mem_alloc(frame_size)
        self.cur_frames_gpu = cuda.mem_alloc(frame_size * frame_cnt)
        self.tmp_gpu = cuda.mem_alloc(frame_size * frame_cnt * 4)
        self.input_size_gpu = cuda.mem_alloc(2 * 4)
        self.RMSE_gpu = cuda.mem_alloc(frame_cnt * 4)

        self.RMSE = np.zeros(frame_cnt, np.int32)

    def most_matching_frame(self, lst_frame, cur_frames):
        # 记录当前帧尺寸，将当前帧展开为一维向量
        input_size = np.array(cur_frames.shape)
        cur_frames = cur_frames.reshape(cur_frames.shape[0] * cur_frames.shape[1])

        # 拷贝数据
        cuda.memcpy_htod(self.lst_frame_gpu, lst_frame)
        cuda.memcpy_htod(self.cur_frames_gpu, cur_frames)
        cuda.memcpy_htod(self.input_size_gpu, input_size)

        # 执行程序
        self.func(self.lst_frame_gpu, self.cur_frames_gpu, self.tmp_gpu, self.RMSE_gpu,
                  self.input_size_gpu, block=(1024, 1, 1), grid=(self.frame_cnt, 1))

        # 拷贝结果误差平方和
        cuda.memcpy_dtoh(self.RMSE, self.RMSE_gpu)
        return self.RMSE

    def test(self, frame_size, frame_cnt):
        lst_frame = np.zeros(frame_size, np.uint8)
        for i in range(frame_size):
            lst_frame[i] = 1

        cur_frames = np.zeros((frame_cnt, frame_size), np.uint8)
        RMSE = self.most_matching_frame(lst_frame, cur_frames)
        print(RMSE)

if __name__ == '__main__':
    frame_size = 256 * 256
    frame_cnt = 100
    c = CoilOffset(frame_size, frame_cnt)
    c.test(frame_size, frame_cnt)