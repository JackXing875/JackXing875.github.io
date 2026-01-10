---
layout: archive
title: "简历"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

教育背景
======
* **中国人民大学**  
  高瓴人工智能学院
  专业方向：人工智能 
  时间：2024 – 至今

---

项目与课程报告
======
* **全连接神经网络运行机制**  
  * 课程：人工智能导论  
  * 内容：前向传播、反向传播、损失函数与参数更新  
  * 实现：基于 NumPy 的多层感知机

* **CacheLab 实验报告**  
  * 课程：操作系统  
  * 内容：Cache 映射方式、命中率分析与示例推导

---

技能
======
* 编程语言：
  * C++
  * Python
  * Java
* 工具与环境：
  * Linux
  * Git / GitHub
  * VS Code
* 相关方向：
  * 机器学习 / 深度学习
  * SLAM

---

出版物（课程报告）
======
<ul>
{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}
</ul>
