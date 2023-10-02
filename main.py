import base64
import pickle
import uvicorn
import torch, cv2
import io, os, time
import numpy as np
import random, base64
from PIL import Image
from yolo import YOLO, YOLO_ONNX
from starlette.responses import FileResponse
from yolo_utils.logger import log, setup_logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Response

# 禁用ReDoc
app = FastAPI(
    title="YOLOv7 SJ",
    description="YOLOv7基于FastAPI构建后端 @todrives",
    redoc_url=None,
    version="1.2.0",
)
# YOLOV7网络
yolo = YOLO()
# 日志记录器
setup_logging()


# ---------------------------------------------------------#
#   解决跨域资源调用问题
# ---------------------------------------------------------#


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------#
#   公共函数部分
# ---------------------------------------------------------#


def temp_file() -> str:
    """
    取当前时间并格式化,用于文件夹命名
    """
    return str(time.strftime("%Y_%m_%d", time.localtime()))


def temp_files() -> str:
    """
    临时文件名生成,时间戳+乱序字母
    """
    random_str = list("abcdefg")
    random.shuffle(random_str)
    return "{}_{}".format(time.time(), random_str)


def dir_isexist(path: str) -> None:
    """
    检测文件夹是否存在,不存在则创建
    """
    if not os.path.exists(path):
        os.mkdir(path)


def getpic_base64(image_path) -> str:
    """
    读取本地图片并base64编码 [base64:str]
    """
    with open(image_path, "rb") as img_obj:
        base64_data = base64.b64encode(img_obj.read())
    return base64_data


def bytes_tobase64(images) -> str:
    """
    图片二进制base64编码 [二进制]->[base64:str]\n
    图片转byte,image: 必须是PIL格式,image_bytes: 二进制
    """
    # 创建一个字节流管道
    img_bytes = io.BytesIO()
    # 将图片数据存入字节流管道, format可以按照具体文件的格式填写
    images.save(img_bytes, format="JPEG")
    # 从字节流管道中获取二进制
    image_bytes = img_bytes.getvalue()
    # 使用base64进行加密
    base64_data = str(base64.b64encode(image_bytes), "utf-8")
    return base64_data


# ---------------------------------------------------------#
#   @/
# ---------------------------------------------------------#
@app.get("/", tags=["根目录访问"])
async def read_root():
    """
    成功访问会返回json {'Hello':'SJU'}
    """
    return {"Hello": "SJU"}


# ---------------------------------------------------------#
#   @predict
# ---------------------------------------------------------#
@app.post("/predict", tags=["图片/视频帧推理"])
async def predict(file: UploadFile):
    """
    @predict 单张图片/视频帧预测\n
    Input:图片\n
    Output:Json
    """
    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Data failed to open properly")
    # 输入到网络接口预测
    image, img_dic = yolo.detect_image(image)
    image_base64 = bytes_tobase64(image)
    # 拼接Json
    if len(img_dic):
        classes = ""
        # dic[标签]=[数量,得分,ymin,xmin,ymax,xmax]
        for key, value in img_dic.items():
            temp = ""
            log(f"key:{key},value:{value}", 4)
            for i in range(value[0]):
                temp += f"""
                "{i}":[{{
                    "score":{value[1+(i*5)]},
                    "ymin":{value[2+(i*5)]},
                    "xmin":{value[3+(i*5)]},
                    "ymax":{value[4+(i*5)]},
                    "xmax":{value[5+(i*5)]} 
                }}],"""
            temp = temp[:-1]
            classes += f'{{"label":"{key}","number":{value[0]},"list":[{{{temp}}}]}},'
        # 去掉末尾的,
        classes = classes[:-1]
    else:
        classes = ""
    dict_json = f'{{"classes": [{classes}], "image": "{image_base64}"}}'

    return Response(content=dict_json, media_type="application/json")


# ---------------------------------------------------------#
#   @track
# ---------------------------------------------------------#
# @app.post("/tracker", tags=["ByteTrack目标追踪"])
# async def tracker(file: UploadFile):
#     """
#     @track ByteTrack跟踪\n
#     Input:视频帧\n
#     Output:Json
#     """
#     try:
#         image = Image.open(file.file)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Data failed to open properly")
#     # 将接收到的列表分割成原列表
#     # item = [i.strip() for i in items[0].split(",")]

#     # 输入到网络接口预测
#     image, img_dic = yolo.detect_track(image)
#     image_base64 = bytes_tobase64(image)
#     # 拼接Json
#     if len(img_dic):
#         classes = ""
#         # dic[标签]=[数量,得分,ymin,xmin,ymax,xmax]
#         for key, value in img_dic.items():
#             temp = ""
#             log(f"key:{key},value:{value}", 4)
#             # 跳过没有追踪物体
#             if len(value) == 1:
#                 continue
#             for i in range(value[0]):
#                 temp += f"""
#                 "{i}":[{{
#                     "id":{value[1+(i*6)]},
#                     "score":{value[2+(i*6)]},
#                     "ymin":{value[3+(i*6)]},
#                     "xmin":{value[4+(i*6)]},
#                     "ymax":{value[5+(i*6)]},
#                     "xmax":{value[6+(i*6)]}  
#                 }}],"""
#             temp = temp[:-1]
#             classes += f'{{"label":"{key}","number":{value[0]},"list":[{{{temp}}}]}},'
#         # 去掉末尾的,
#         classes = classes[:-1]
#     else:
#         classes = ""

#     dict_json = f'{{"classes": [{classes}], "image": "{image_base64}"}}'

#     return Response(content=dict_json, media_type="application/json")


# ---------------------------------------------------------#
#   @fps
# ---------------------------------------------------------#


@app.post("/fps", tags=["FPS测试"])
async def fps(file: UploadFile, test_interval: int = 10):
    """
    @fps 图片推理fps测试\n
    测试次数 test_interval int\n
    理论上test_interval越大,fps越准确
    """
    img_file = file.file
    try:
        image = Image.open(img_file)
    except Exception:
        log("@fps 打开图片失败", 1)
        raise HTTPException(status_code=400, detail="Data failed to open properly")
    else:
        tact_time = yolo.get_FPS(image, test_interval)

    return {"seconds": tact_time, "fps": 1 / tact_time}


# ---------------------------------------------------------#
#   @heatmap
# ---------------------------------------------------------#


@app.post("/heatmap", tags=["热力图"])
async def heatmap(file: UploadFile = File(...)):
    """
    @heatmap\n
    输出热力图,base64编码
    """
    imagefile = file.file
    try:
        image = Image.open(imagefile)
    except Exception:
        log("@heatmap 打开图片失败", 1)
        raise HTTPException(status_code=400, detail="Data failed to open properly")
    else:
        # 推理并保存热力图
        image = yolo.detect_heatmap(image)
        image_base64 = bytes_tobase64(image)
    return {"image": image_base64}


# @ app.post('/export_onnx', tags=['模型导出onnx'])
# async def export_onnx(simplify: bool = True, onnx_save_path: str = 'model_data\models.onnx'):
#     '''
#     @export_onnx 模型导出为onnx,需要pytorch1.7.1以上\n
#     simplify            使用Simplify onnx\n
#     onnx_save_path      指定了onnx的保存路径,默认'model_data\models.onnx'
#     '''
#     #yolo.convert_to_onnx(simplify, onnx_save_path)
#     return {'filename': simplify}


# ---------------------------------------------------------#
#   @camera_ca1
# ---------------------------------------------------------#


@app.post("/camera_ca1", tags=["格图相机标定"])
async def camera_calibration1(
    w: int = 9,
    h: int = 6,
    chesslength: int = 25,
    imagefiles: list[UploadFile] = File(...),
):
    """
    @camera_calibration 使用网格图进行相机标定\n
    棋盘格模板规格 棋盘格内角点\n
    w = 9(10-1)\n
    h = 6(7-1)\n
    chesslength = 100   每个棋盘小格的实际长度,单位为mm\n
    imagefiles   传入标定的图片(一般为15-20张图片,角度多样)\n
    返回保存的标定数据,格子图占相机图幅1/4,在各个角落
    """
    # 计算亚像素角点时终止迭代阈值,最大计算次数30次,最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 准备格式如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)的3d角点
    objp = np.zeros((w * h, 3), np.float32)
    # 用于存储来自所有图像的对象点和图像点的数组
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # 存储3D角点
    objpoints = []
    # 存储2D角点
    imgpoints = []
    for fname in imagefiles:
        temp_path = "./output_logs/temp/{}.png".format(temp_files())
        # 打开图片
        img = Image.open(fname.file)
        # 将图片保存
        img.save(temp_path)
        # 读取图片的格式为[h,w,c]
        img = cv2.imread(temp_path)
        # 删除缓存图片
        os.remove(temp_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        log("ret:{}".format(ret), 4)
        if ret == True:
            objpoints.append(objp * chesslength)
            # 角点精细化,其中corners为初始计算的角点向量
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # 绘制角点并展示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.imshow("img", img)
            cv2.waitKey(0)
        # 释放系统资源
    cv2.destroyAllWindows()

    # 相机标定,依次返回标定结果、内置参数矩阵、畸变参数、旋转向量、平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None
    )
    print("INFO : @camera_calibration ret:", ret)  # ret为bool值
    print("INFO : @camera_calibration mtx:\n", mtx)  # 内参数矩阵
    # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("INFO : @camera_calibration dist:\n", dist)
    print("INFO : @camera_calibration rvecs:\n", rvecs)  # 旋转向量,外参数
    print("INFO : @camera_calibration tvecs:\n", tvecs)  # 平移向量,外参数

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    # 将参数保存到本地
    params = {"camera_matrix": mtx, "dist_coeffs": dist[0:4]}
    with open("calibration_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("标定参数已保存至文件:calibration_params.pkl")

    # return {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    return FileResponse("calibration_params.pkl", filename="calibration_params.pkl")


# ---------------------------------------------------------#
#   @corrects_dis
# ---------------------------------------------------------#


@app.post("/corrects_dis", tags=["纠正畸变"])
async def corrects_distortions(file: UploadFile = File(...)):
    """
    @corrects_distortions 纠正畸变\n
    file 需要纠正畸变的图像
    """
    try:
        image = Image.open(file)
    except Exception:
        log("@heatmap 打开图片失败", 1)
        raise HTTPException(status_code=400, detail="Data failed to open properly")

    with open("calibration_params.pkl", "rb") as f:
        params = pickle.load(f)
    camera_matrix, dist_coeffs = params["camera_matrix"], params["dist_coeffs"]
    # 进行畸变修正
    undistorted_frame = cv2.undistort(image, camera_matrix, dist_coeffs)
    return {"encoded_img": bytes_tobase64(undistorted_frame)}


# ---------------------------------------------------------#
#   @info
# ---------------------------------------------------------#


@app.get("/info", tags=["服务器环境"])
async def info():
    """@info 返回运行环境基础信息"""
    return {
        "torchversion": torch.__version__,
        "CUDAversion": torch.version.cuda,
        "CUDAstate": torch.cuda.is_available(),
    }


# ---------------------------------------------------------#
#   主函数入口
# ---------------------------------------------------------#
if __name__ == "__main__":
    # 热力图文件夹
    dir_isexist("./output_logs/heatmap/")
    # 推理视频文件夹
    dir_isexist("./output_logs/video/")
    # 截取图片文件夹
    dir_isexist("./output_logs/image/")
    # 临时文件文件夹
    dir_isexist("./output_logs/temp/")
    # 启动服务器
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
