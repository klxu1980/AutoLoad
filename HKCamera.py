import time
import json
import sys 
from ctypes import *
import os
import numpy as np

from Camera import Camera
import cv2
import configparser

#sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
sys.path.append("D:/MVS/Development/Samples/Python/MvImport")
from MvCameraControl_class import *


class HKCamera(Camera):
    def __init__(self, CameraIdx=0, log_path=None):
        self.camera = None

        # enumerate all the camera devices
        # the Exception should be handled by its involker
        deviceList = self.enum_devices()

        # generate a camera instance
        self.camera = self.open_camera(deviceList, CameraIdx, log_path)
        self.start_camera()

    def __del__(self):
        if self.camera is None:
            return

        # 停止取流
        ret = self.camera.MV_CC_StopGrabbing()
        if ret != 0:
            raise Exception("stop grabbing fail! ret[0x%x]" % ret)

        # 关闭设备
        ret = self.camera.MV_CC_CloseDevice()
        if ret != 0:
            raise Exception("close deivce fail! ret[0x%x]" % ret)

        # 销毁句柄
        ret = self.camera.MV_CC_DestroyHandle()
        if ret != 0:
            raise Exception("destroy handle fail! ret[0x%x]" % ret)

    @staticmethod
    def enum_devices():
        """
        枚举网口、USB口、未知设备、cameralink 设备
        """
        cameraType = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_UNKNOW_DEVICE | MV_1394_DEVICE | MV_CAMERALINK_DEVICE
        deviceList = MV_CC_DEVICE_INFO_LIST()
        # 枚举设备
        ret = MvCamera().MV_CC_EnumDevices(cameraType, deviceList)
        if ret != 0:
            raise Exception("枚举海康相机失败! ret[0x%x]" % ret)
        return deviceList

    def open_camera(self, deviceList, CameraIdx, log_path):
        # generate a camera instance
        camera = MvCamera()

        # 选择设备并创建句柄
        stDeviceList = cast(deviceList.pDeviceInfo[CameraIdx], POINTER(MV_CC_DEVICE_INFO)).contents
        if log_path is not None:
            ret = self.camera.MV_CC_SetSDKLogPath(log_path)
            if ret != 0:
                raise Exception("set Log path  fail! ret[0x%x]" % ret)

            # 创建句柄,生成日志
            ret = camera.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                raise Exception("create handle fail! ret[0x%x]" % ret)
        else:
            # 创建句柄,不生成日志
            ret = camera.MV_CC_CreateHandleWithoutLog(stDeviceList)
            if ret != 0:
                raise Exception("create handle fail! ret[0x%x]" % ret)

        # 打开相机
        ret = camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise Exception("open device fail! ret[0x%x]" % ret)

        return camera

    def start_camera(self):
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.camera.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise Exception("get payload size fail! ret[0x%x]" % ret)

        self.nDataSize = stParam.nCurValue
        self.pData = (c_ubyte * self.nDataSize)()
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))

        self.camera.MV_CC_StartGrabbing()

    def get_Value(self, param_type, node_name):
        """
        :param cam:            相机实例
        :param_type:           获取节点值得类型
        :param node_name:      节点名 可选 int 、float 、enum 、bool 、string 型节点
        :return:               节点值
        """
        if param_type == "int_value":
            stParam = MVCC_INTVALUE_EX()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
            ret = self.camera.MV_CC_GetIntValueEx(node_name, stParam)
            if ret != 0:
                raise Exception("获取 int 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            return stParam.nCurValue

        elif param_type == "float_value":
            stFloatValue = MVCC_FLOATVALUE()
            memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.camera.MV_CC_GetFloatValue(node_name, stFloatValue)
            if ret != 0:
                raise Exception("获取 float 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            return stFloatValue.fCurValue

        elif param_type == "enum_value":
            stEnumValue = MVCC_ENUMVALUE()
            memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
            ret = self.camera.MV_CC_GetEnumValue(node_name, stEnumValue)
            if ret != 0:
                raise Exception("获取 enum 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            return stEnumValue.nCurValue

        elif param_type == "bool_value":
            stBool = c_bool(False)
            ret = self.camera.MV_CC_GetBoolValue(node_name, stBool)
            if ret != 0:
                raise Exception("获取 bool 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            return stBool.value

        elif param_type == "string_value":
            stStringValue = MVCC_STRINGVALUE()
            memset(byref(stStringValue), 0, sizeof(MVCC_STRINGVALUE))
            ret = self.camera.MV_CC_GetStringValue(node_name, stStringValue)
            if ret != 0:
                raise Exception("获取 string 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            return stStringValue.chCurValue

        else:
            return None

    def set_Value(self, param_type, node_name, node_value):
        """
        :param cam:               相机实例
        :param param_type:        需要设置的节点值得类型
            int:
            float:
            enum:     参考于客户端中该选项的 Enum Entry Value 值即可
            bool:     对应 0 为关，1 为开
            string:   输入值为数字或者英文字符，不能为汉字
        :param node_name:         需要设置的节点名
        :param node_value:        设置给节点的值
        :return:
        """
        if param_type == "int_value":
            ret = self.camera.MV_CC_SetIntValueEx(node_name, int(node_value))
            if ret != 0:
                raise Exception("设置 int 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))

        elif param_type == "float_value":
            ret = self.camera.MV_CC_SetFloatValue(node_name, float(node_value))
            if ret != 0:
                raise Exception("设置 float 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))

        elif param_type == "enum_value":
            ret = self.camera.MV_CC_SetEnumValue(node_name, node_value)
            if ret != 0:
                raise Exception("设置 enum 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))

        elif param_type == "bool_value":
            ret = self.camera.MV_CC_SetBoolValue(node_name, node_value)
            if ret != 0:
                raise Exception("设置 bool 型数据节点 %s 失败 ！ 报错码 ret[0x%x]" % (node_name, ret))

        elif param_type == "string_value":
            ret = self.camera.MV_CC_SetStringValue(node_name, str(node_value))
            if ret != 0:
                raise Exception("设置 string 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))

    def set_exposure_time(self, exp_time):
        self.set_Value(param_type="float_value", node_name="ExposureTime", node_value=exp_time)

    def get_exposure_time(self):
        return self.get_Value(param_type="float_value", node_name="ExposureTime")

    def get_image(self, width=None):
        """
        :param cam:     相机实例
        :active_way:主动取流方式的不同方法 分别是（getImagebuffer）（getoneframetimeout）
        :return:
        """
        ret = self.camera.MV_CC_GetOneFrameTimeout(self.pData, self.nDataSize, self.stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(self.pData).reshape((self.stFrameInfo.nHeight, self.stFrameInfo.nWidth))
            if width is not None:
                image = cv2.resize(image, (width, int(self.stFrameInfo.nHeight * width / self.stFrameInfo.nWidth)))
                pass
            return image
        else:
            return None

    def show_runtime_info(self, exp_time,image):
        #exp_time = self.get_exposure_time()
        temp1 = float(os.popen("cat /sys/devices/virtual/thermal/thermal_zone0/temp").read())/1000
        temp2 = float(os.popen("cat /sys/devices/virtual/thermal/thermal_zone2/temp").read())/1000
        cv2.putText(image, ("exposure time = %1.1fms" % (exp_time * 0.001)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(image,("cputemp = %6.1f C" %(temp1)), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,255, 1)
        cv2.putText(image,("gputemp = %6.1f C" %(temp2)), (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,255, 1)


if __name__ == '__main__':
    camera = HKCamera()
    while True:
        img = camera.get_image()
        cv2.imshow("", img)
        cv2.waitKey(100)

    """
    #     ftp = FTPService(server_ip=config.get("information","server_ip"), server_port=int(config.get("information","server_port")), username=config.get("information","username"), password=config.get("information","password"))
        htp=http_service(url='http://49.232.164.34/api/upload/')

        client_id = '20baa6e650114ef7a9bebd864f4e8db6'  # MQTT 的 ClientID，需要保证全局唯一
        # broker_url = 'broker.zxj.runxinhangkong.cn'
        broker_url = '49.232.164.34'
        port = 9883
        username = 'zxj_v2_dev'
        password = 'dfefjklg3k535kjk3kdsd83hjfr'

        mc = MyClient(client_id, broker_url, port, username, password, tls=False)
        global x, y, wide, height
        x = 1104
        y = 621
        height = 500
        wide = 500
        # 订阅topic样例，页面刷新订阅
        rtn0 = mc.subscribe(f'zxj/v2/{client_id}/read')
        # 订阅topic样例，设置信息订阅
        rtn1 = mc.subscribe(f'zxj/v2/{client_id}/write')



        exp_time = camera.auto_exposure_time(min_exp_time=float(config.get("parameter","min_exp_time")), max_exp_time=float(config.get("parameter","max_exp_time")))
        count=1
        try:

            while True:
                    time.sleep(15)
                    status = camera.set_led(exp_time, flag=flag)

                    if status == 'NORMAL':
                        if count == 500:
                            count = 1
                            exp_time = camera.auto_exposure_time(min_exp_time=float(config.get("parameter","min_exp_time")), max_exp_time=float(config.get("parameter","max_exp_time")))
                        list_image=camera.gain_image(repeat_cnt=10,width=800)
                        clr_image = camera.get_clearest_image1(list_image=list_image,repeat_cnt=10)
                        up_file_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime())
                        local_file_name = "/dev/shm/" + up_file_name
                        cv2.imwrite(local_file_name, clr_image)
                        information = htp.http_upload(files={'file': open(local_file_name, 'rb')})
                        information = json.loads(information)
                        flag1 = False
                        # 发送消息的样例
                        dict1 = {"timestamp":time.localtime(),"data":{"df1":information['file'],"df2":exp_time,"df3":flag1}}
                        json_info = json.dumps(dict1)
                        rc = mc.publish(f'zxj/v2/{client_id}/data', json_info)

                        # print('upload sucess')
        #                 ftp.upload(local_file_name, up_file_name)
        #             print("success upload 2")
                        os.remove(local_file_name)
                        count += 1

                    if status == 'OVER':
                        gpio.mode_set(flag=0)
                        exp_time = camera.auto_exposure_time(min_exp_time=float(config.get("parameter","min_exp_time")), max_exp_time=float(config.get("parameter","max_exp_time")))
                        list_image=camera.gain_image(repeat_cnt=10,width=800)
                        clr_image = camera.get_clearest_image1(list_image=list_image,repeat_cnt=10)
                        time.sleep(1)
                        up_file_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime())
                        local_file_name = "/dev/shm/" + up_file_name
                        cv2.imwrite(local_file_name, clr_image)
                        information = htp.http_upload(files={'file': open(local_file_name, 'rb')})
                        information = json.loads(information)
                        flag1 = False
                        dict1 = {"timestamp":time.localtime(),"data":{"df1":information['file'],"df2":exp_time,"df3":flag1}}
                        json_info = json.dumps(dict1)
                        rc = mc.publish(f'zxj/v2/{client_id}/data', json_info)

                        #print('upload sucess')
        #                 ftp.upload(local_file_name, up_file_name)
        #             print("success upload 1")
                        os.remove(local_file_name)

                    if status == 'UNDER':
                        flag1 = True
                        gpio.mode_set(flag=1)
                        exp_time = camera.auto_exposure_time(min_exp_time=float(config.get("parameter","min_exp_time")), max_exp_time=float(config.get("parameter","max_exp_time")))
                        list_image=camera.gain_image(repeat_cnt=10,width=800)
                        time.sleep(1)
                        gpio.mode_set(flag=0)
                        clr_image = camera.get_clearest_image1(list_image=list_image,repeat_cnt=10)

                        up_file_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime())
                        local_file_name = "/dev/shm/" + up_file_name
                        cv2.imwrite(local_file_name, clr_image)

                        information = htp.http_upload(files={'file': open(local_file_name, 'rb')})
                        information = json.loads(information)
                        # 订阅topic样例
                        # mc.subscribe(f'zxj/v2/{client_id}/read')
                        # 发送消息的样例
                        dict1 = {"timestamp":time.localtime(),"data":{"df1":information['file'],"df2":exp_time,"df3":flag1}}
                        json_info = json.dumps(dict1)
                        rc = mc.publish(f'zxj/v2/{client_id}/data', json_info)

        #                 print('upload sucess')
        #                 ftp.upload(local_file_name, up_file_name)
                        os.remove(local_file_name)
                    gpio1.mode_set(flag=1)
                    camera.show_runtime_info(exp_time,clr_image)
                    cv2.imshow("", clr_image)
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('e') or key == ord('E'):
                        cv2.destroyAllWindows()
                        break
        except Exception as e:
           gpio1.mode_set(flag=0)
           print(e)
   except Exception as ex:
       print(ex)
    """