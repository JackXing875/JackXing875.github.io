---
layout: academic
title: "Curriculum vitae"
permalink: /cv/
page_class: cv-page
subtitle: "Tiancheng Xing · Yuu"
redirect_from:
  - /resume
  - /cv-json/
  - /resume-json
---

<div class="cv-toolbar">
  <p><a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a><span aria-hidden="true"> · </span><a href="https://github.com/{{ site.author.github }}">GitHub</a><span aria-hidden="true"> · </span><a href="{{ site.author.googlescholar }}">Google Scholar</a></p>
  <button type="button" class="outline-button" onclick="window.print()">Print / Save PDF</button>
</div>

## Education

<div class="cv-entry" markdown="1">
<p class="cv-date">2024 – present</p>
<div markdown="1">

### Renmin University of China

**B.S. candidate in Artificial Intelligence**<br />
Gaoling School of Artificial Intelligence · Beijing, China

</div>
</div>

## Research interests

Multimodal reasoning; efficient multimodal inference; video generation; embodied intelligence. Current interests include KV cache compression, token pruning, sparse attention, adaptive caching, and vision-language models.

## Publications

{% for paper in site.data.research %}
  {% include research-publication.html paper=paper compact=true %}
{% endfor %}

## Selected projects

{% for project in site.data.projects %}
<div class="cv-project">
  <h3><a href="{{ project.url }}">{{ project.name }} <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></h3>
  <p>{{ project.description }}</p>
  <p class="project-tools">{{ project.tools }}</p>
</div>
{% endfor %}

## Technical skills

- **Programming:** Python, C++, Java.
- **AI & perception:** Retrieval-augmented generation, FAISS semantic retrieval, visual SLAM, and dense reconstruction.
- **Development:** Linux, Git, FastAPI, and Vue 3.

## Selected coursework & reports

- **Introduction to Artificial Intelligence:** [Course notes]({{ '/publication/AI_intro' | relative_url }}), including a NumPy implementation of a multilayer perceptron.
- **Introduction to Computer Systems:** [LinkLab]({{ '/publication/linklab_report' | relative_url }}), [AttackLab]({{ '/publication/attacklab_report' | relative_url }}), [CacheLab]({{ '/publication/cachelab_report' | relative_url }}), [BombLab]({{ '/publication/bomblab_report' | relative_url }}), and [DataLab]({{ '/publication/datalab_report' | relative_url }}).
