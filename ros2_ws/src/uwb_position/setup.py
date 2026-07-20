from setuptools import setup

package_name = 'uwb_position'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/uwb_position/launch', ['launch/fusion_launch.py', 'launch/bringup_launch.py']),
        ('share/uwb_position/config', ['config/ekf.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carpe',
    maintainer_email='carpe@todo.todo',
    description='UWB trilateration with live 2D plot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uwb_position_node = uwb_position.uwb_position_node:main',
            'uwb_vehicule_node = uwb_position.uwb_vehicule:main',
            'uwb_to_odom = uwb_position.uwb_to_odom:main',
            'fused_odom_node=uwb_position.fused_odom_node:main',
            'eskf_node=uwb_position.eskf_node:main',
            'nlos_context_node=uwb_position.nlos_context_node:main'
        ],
    },
)
