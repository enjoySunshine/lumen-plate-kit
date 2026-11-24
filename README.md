lumen-plate-kit

构建静态库与页面生成工具：

    make

若环境里的默认 cc 不可用，可显式指定编译器：

    make CC=gcc

生成 view/index.html 所需的全部图片与计时数据（增益图、三档曝光预览、
CPU 时间与内容光级）：

    make view

然后在浏览器中打开 view/index.html。页面上的每张图和旁边的毫秒数都来自
同一次运行：先做一次不计入的空跑，再连跑四次，页面展示这四次的最小值和
最大值（用户态 + 内核态 CPU 时间）。
