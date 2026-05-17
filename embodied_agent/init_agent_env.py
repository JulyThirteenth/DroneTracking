
import os, math
from dotenv import load_dotenv

def init_agent_env():
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "OPENAI").upper()

    if not (api_key := os.getenv("LLM_API_KEY")):
        raise ValueError(f"Error: 未检测到 LLM_API_KEY，请检查 .env 文件或环境变量是否配置正确。")

    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))

    if provider == "OPENAI":
        from langchain_openai import ChatOpenAI
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(
            model=model_name or "gpt-4o",
            base_url=base_url if base_url else None,
            temperature=temperature
        )
    elif provider == "GOOGLE":
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-pro",
            base_url=base_url if base_url else None,
            temperature=temperature
        )
    else:
        raise ValueError(f"Error: Unknown API provider")

    print(f"{provider}模型 {model_name} 加载成功, TEMP={temperature}")

    from .ros2_control import ROS2Control
    from .spf_tools import init_env
    from .spf_geometry import SPFGeometry

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

    init_env(
        control=ROS2Control(spf_geometry=SPFGeometry(width, height, hfov, vfov)),
        sub_llm=llm
    )

    print("智能体已与ROS2环境及大模型成功绑定")