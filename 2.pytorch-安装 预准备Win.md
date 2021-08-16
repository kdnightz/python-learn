# pytorch安装 预准备

1. 安装anaconda

   1. https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 清华源版本

   2. next-i agree-justme-固态充足选固态，否则选择容量大些的盘-双勾-都不打钩-结束

   3. 左下角开始-打开 Anaconda3-Anacaonda prompt，输入

      1. `conda config --set show_channel_urls yes`

   4. C:\Users\用户名 找到 .condarc 右键编辑,全删除并改为以下

      1. ```yaml
         channels:
           - defaults
         show_channel_urls: true
         default_channels:
           - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
           - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
           - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
         custom_channels:
           conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
           msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
           bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
           menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
           pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
           simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
         ```

      2. 虚拟环境
         进入 Anacaonda prompt

         ```
         activate // 切换到base环境
         
         activate learn // 切换到learn环境
         
         conda create -n learn python=3 // 创建一个名为learn的环境并指定python版本为3(的最新版本)
         
         conda env list // 列出conda管理的所有环境
         
         conda list // 列出当前环境的所有包
         
         conda install requests 安装requests包
         
         conda remove requests 卸载requets包
         
         conda remove -n learn --all // 删除learn环境及下属所有包
         
         conda update requests 更新requests包
         
         conda env export > environment.yaml // 导出当前环境的包信息
         
         conda env create -f environment.yaml // 用配置文件创建新的虚拟环境
         ```

         1. 首先 输入 

            conda create -n learn python=3 // 创建一个名为learn的环境并指定python版本为3(的最新版本)

         2. 然后 

            activate learn // 切换到learn环境

         3. 之后在learn环境下进行操作



# CUDA

1. 第一步：查看自己是否有支持安装CUDA的NVIDA显卡

   1. 控制面板-NVIDA控制面板-帮助-系统信息-组件

      看到NVCUDA.DLL 后看产品名称 显示是什么 例如10.2.xxx就是10.2版本

2. 第二步：查看是否有NVIDA显卡驱动程序，如果有，就不用安装了（一般刚装完系统都会安装这些驱动），建议自动更新驱动程序一下，没有，请下载安装，地址：https://www.geforce.cn/drivers，有两种安装方式，自动和手动，选择适合自己电脑的显卡驱动下载，安装很简单，直接下一步就可以，默认系统安装路径。

   如果你游戏玩家，最好不要驱动盲目升到最新。

3. 下载CUDA

   1. 第一步：到官网下载CUDA安装包，前面我们已经查看到了电脑GPU显卡所支持的CUDA版本为10.2,下载地址：https://developer.nvidia.com/cuda-toolkit-archive

      选择 windows-10-local-base installer 右边download。

4. 接着就是安装过程，双击打开显示临时解压目录，不需要改变，默认即可。记住位置，事后删除即可

   ​     接下来，进入NVIDIA安装过程，在这安装过程中，自定义安装，把cuda-VS intergration 去掉。安装完毕。

5. cmd 里 `nvcc -V`输入后，返回版本，安装成功。



# CUDNN

1. https://developer.nvidia.com/rdp/cudnn-archive

2. 先注册一个账号，然后完成一个问卷。

3. 选择适配CUDA的版本，如我们是10.2的CUDA，则下载`Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 10.2` 并选择[cuDNN Library for Windows10 (x86)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/10.2_06072021/cudnn-10.2-windows10-x64-v8.2.1.32.zip)

4. 下载之后，解压缩，将CUDNN压缩包里面的bin、clude、lib文件直接复制到CUDA的安装目录下，直接覆盖安装即可。

   ```
   1.  C:\cuda\bin\cudnn64_7.dll —> C:\Program Files\NVIDIA GPUComputing 
   Toolkit\CUDA\v9.1\bin
   
   2.  C:\cuda\include\cudnn.h —> C:\Program Files\NVIDIA GPUComputing 
   Toolkit\CUDA\v9.1\include
   
   3.  C:\cuda\lib\x64\cudnn.lib —> C:\Program Files\NVIDIA GPUComputing 
   Toolkit\CUDA\v9.1\lib\x64
   ```



# Pytorch

1. https://pytorch.org/get-started/locally/

2. 选择 stable-windows-conda-python-cuda10.2 然后

   ```
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```

   这段话输入在 learn环境下的 conda prompt中。安装成功

3. 输入python 进入py解释器下

4. 输入 `import torch`

5. 输入 `torch.__version__`，返回版本号

6. 输入`torch.cuda.is_available()` 返回true

7. 成功