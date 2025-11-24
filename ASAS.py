# Inspect_GC.py
from VeraGridEngine import api as gce

print("正在检查 GridCal Bus 对象属性...")
try:
    # 尝试创建一个测试 Bus
    b = gce.Bus(name="TestBus")

    # 打印它所有的属性和方法
    print("\n[Bus 对象的属性列表]:")
    attributes = dir(b)
    public_attrs = [a for a in attributes if not a.startswith('__')]
    print(public_attrs)

    # 尝试打印一些关键属性的当前值
    print("\n[关键属性检查]:")
    for attr in ['vn_kv', 'vnom', 'Vnom', 'vmin', 'Vmin', 'min_v', 'vmax', 'Vmax', 'max_v']:
        if hasattr(b, attr):
            print(f"  ✅ 发现属性: {attr}")
        else:
            print(f"  ❌ 无此属性: {attr}")

except Exception as e:
    print(f"检查时出错: {e}")