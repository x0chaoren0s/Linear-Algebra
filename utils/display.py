def myDisplay(obj):
    """
    根据运行环境自动选择显示方式
    在Jupyter环境中使用display，在普通Python环境中使用print
    """
    try:
        # 尝试导入IPython模块
        from IPython.display import display
        # 如果成功，说明在Jupyter环境中
        display(obj)
    except ImportError:
        # 如果导入失败，说明在普通Python环境中
        print(obj) 