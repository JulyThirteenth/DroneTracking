
import os, math
from dotenv import load_dotenv
from .ros2_control import ROS2Control
from .spf_geometry import SPFGeometry
from .spf_tools import init_env, TOOLS_LIST
from .spf_agent import create_spf_agent
from .task_loader import ModelConfig

def init_ros2_control() -> ROS2Control:
    # 用`ros2 topic echo /rgb --once | grep -E "width:|height:"`来获取 width 和 height
    width = 640
    height = 480

    # 在isaac sim中查看 camera_fpv 属性:
    # Focal Length = 12.0 
    # Horizontal Aperture = 24.0
    # Vertical Aperture  = 18.0

    # hfov = 2 * arctan(24.0 / (2 * 12.0)) = 90°
    hfov = 90.0
    if width != height:
        hfov_rad = math.radians(hfov)
        vfov_rad = 2 * math.atan(math.tan(hfov_rad / 2) * (height / width))
        vfov = math.degrees(vfov_rad)
    else:
        vfov = hfov
    return ROS2Control(spf_geometry=SPFGeometry(width, height, hfov, vfov))

def _create_llm(model_config: ModelConfig):
    load_dotenv()
    provider = model_config.provider
    api_key = model_config.api_key

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_config.model_name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            api_key=api_key
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    print(f"{provider}模型 {model_config.model_name} 加载成功, TEMP={model_config.temperature}")
    return llm

def create_agent(model_config: ModelConfig, control: ROS2Control):
    llm = _create_llm(model_config)
    init_env(control=control, sub_llm=llm)
    print("[embodied_agent] 智能体已与ROS2环境成功绑定")
    return create_spf_agent(llm, tools=TOOLS_LIST)