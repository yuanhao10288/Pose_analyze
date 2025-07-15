import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def calculate_trajectory(ax, ay, az, dt, initial_pos=(0, 0, 0), initial_vel=(0, 0, 0)):
    """
    从三轴加速度数据计算运动轨迹
    
    参数:
    ax, ay, az: 加速度数据 (m/s²)
    dt: 采样时间间隔 (s)
    initial_pos: 初始位置 (x, y, z)
    initial_vel: 初始速度 (vx, vy, vz)
    
    返回:
    轨迹点的坐标列表: [时间, x, y, z, vx, vy, vz, ax, ay, az]
    """
    # 确保所有输入长度一致
    n = len(ax)
    assert len(ay) == n and len(az) == n
    
    # 初始化数组存储轨迹数据
    time = np.arange(0, n * dt, dt)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vz = np.zeros(n)
    
    # 设置初始条件
    x[0], y[0], z[0] = initial_pos
    vx[0], vy[0], vz[0] = initial_vel
    
    # 重力加速度 (假设z轴向上为正)
    g = 9.81  # m/s²
    
    # 数值积分计算轨迹 (欧拉法)
    for i in range(1, n):
        # 考虑重力对z轴加速度的影响
        az_gravity = az[i] - g
        
        # 速度更新 (v = v₀ + a·Δt)
        vx[i] = vx[i-1] + ax[i] * dt
        vy[i] = vy[i-1] + ay[i] * dt
        vz[i] = vz[i-1] + az_gravity * dt
        
        # 位置更新 (s = s₀ + v·Δt)
        x[i] = x[i-1] + vx[i] * dt
        y[i] = y[i-1] + vy[i] * dt
        z[i] = z[i-1] + vz[i] * dt
    
    return np.column_stack((time, x, y, z, vx, vy, vz, ax, ay, az))

def plot_trajectory(trajectory):
    """绘制3D轨迹图和速度变化图"""
    fig = plt.figure(figsize=(14, 6))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], 'b-', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D trajectory of tennis')
    
    # 添加起始和结束标记
    ax1.scatter(trajectory[0, 1], trajectory[0, 2], trajectory[0, 3], color='green', s=50, label='Starting point')
    ax1.scatter(trajectory[-1, 1], trajectory[-1, 2], trajectory[-1, 3], color='red', s=50, label='Ending point')
    ax1.legend()
    
    # 速度随时间变化图
    ax2 = fig.add_subplot(122)
    time = trajectory[:, 0]
    speed = np.sqrt(trajectory[:, 4]**2 + trajectory[:, 5]**2 + trajectory[:, 6]**2)
    ax2.plot(time, speed, 'r-', linewidth=2)
    ax2.set_xlabel('times (s)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_title('Speed varies with time')
    
    # 添加最大速度标记 - 缩短箭头长度
    max_speed_idx = np.argmax(speed)
    # 减少xytext的偏移量（从+0.1和+1改为+0.05和+0.5），使箭头更短
    ax2.annotate(f'max_speed: {speed[max_speed_idx]:.2f} m/s',
                 xy=(time[max_speed_idx], speed[max_speed_idx]),
                 xytext=(time[max_speed_idx], speed[max_speed_idx]),  # 关键修改：减小偏移量
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.show()

def read_acceleration_data(file_path):
    """
    从CSV文件读取加速度数据
    
    参数:
    file_path: CSV文件路径
    
    返回:
    ax, ay, az: 三轴加速度数据 (m/s²)
    dt: 采样时间间隔 (s)
    """
    time_data = []
    ax_data = []
    ay_data = []
    az_data = []
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            
            for row in reader:
                if len(row) >= 4:
                    time_data.append(float(row[0]))
                    ax_data.append(float(row[1]))
                    ay_data.append(float(row[2]))
                    az_data.append(float(row[3]))
        
        # 计算采样时间间隔 (假设为固定间隔)
        if len(time_data) >= 2:
            dt = (time_data[1] - time_data[0]) / 1000.0  # 转换为秒
        else:
            dt = 0.02  # 默认20ms
        
        return np.array(ax_data), np.array(ay_data), np.array(az_data), dt
    
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return np.array([]), np.array([]), np.array([]), 0.0

# 主程序
if __name__ == "__main__":
    file_path = "D:\\desk\\Pose_analyze\\text.csv"  # 请确保CSV文件在正确的路径下
    
    # 读取加速度数据
    ax, ay, az, dt = read_acceleration_data(file_path)
    
    if len(ax) > 0:
        # 计算轨迹
        trajectory = calculate_trajectory(ax, ay, az, dt, 
                                         initial_pos=(0, 0, 1.5),  # 初始位置 (x=0, y=0, z=1.5m)
                                         initial_vel=(0, 0, 0))    # 初始速度为0
        
        # 绘制轨迹图
        plot_trajectory(trajectory)
        
        # 输出关键指标
        max_speed = np.max(np.sqrt(trajectory[:, 4]**2 + trajectory[:, 5]**2 + trajectory[:, 6]**2))
        max_height = np.max(trajectory[:, 3])
        total_distance = np.sum(np.sqrt(np.diff(trajectory[:, 1])**2 + 
                                      np.diff(trajectory[:, 2])**2 + 
                                      np.diff(trajectory[:, 3])**2))
        
        print(f"Analysis results:")
        print(f"Maximum speed: {max_speed:.2f} m/s")
        print(f"Maximum height: {max_height:.2f} m")
        print(f"Total moving distance: {total_distance:.2f} m")
        print(f"Total time: {trajectory[-1, 0]:.2f} s")
    else:
        print("无法读取加速度数据，请检查CSV文件格式和路径。")    