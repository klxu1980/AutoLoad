# 带卷自动上料视觉跟踪程序
## 接口程序
AutoLoad.py中包含了类AutoLoader，是带卷自动跟踪程序的接口。带卷跟踪包括了以下阶段：

1. 开始新带卷的上料跟踪，调用函数init_one_cycle()，初始化一个完整上料周期

2. 带卷位于上料等待区，等待被搬运。调用函数wait_for_loading()，检测带卷在上料区的初始位置。带卷初始位置仅包括带卷内椭圆的中心位置，椭圆的长短轴和角度不进行检测，但会设置一个简单的默认值。wait_for_loading()会返回带卷的初始中心位置(x, y)。

3. 上料跟踪，调用函数uploading()。函数跟踪带卷内圆，返回带卷内椭圆参数(x, y, long, short, angle)，以及在该位置时预期的标准椭圆参数。控制程序可以对比这两个椭圆，判断带卷跟踪是否有异常。

4. 带卷对准并插入开卷机后，判断带卷是否已经完全插入开卷机，调用函数is_coil_into_unpacker()。函数返回bool值，True表示已经完全插入开卷机，可以开卷。

5. 带卷插入开卷机后，判断是否已经开卷，调用函数is_coil_unpacked()。函数返回bool值，True表示已经开卷。

## 设置模型文件
AutoLoader的构造函数中包含了带卷跟踪器coil_tracer，带卷初始位置检测器coil_locator，上卷检测器coil_pos_status，开卷检测器coil_open_status。每个实例均需要通过load_model()函数加载模型文件。如果使用了新模型，注意修改这里。
