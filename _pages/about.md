---
layout: academic
title: "About"
heading: "Tiancheng Xing"
alias: "Yuu"
permalink: /
page_class: home-page
subtitle: "Undergraduate · Artificial Intelligence · Renmin University of China"
redirect_from:
  - /about/
  - /about.html
---

<div class="home-introduction" markdown="1">
<aside class="home-profile" aria-label="Profile">
  <img src="{{ '/images/profile.png' | relative_url }}" alt="Yuu's profile avatar" width="220" height="220" fetchpriority="high" />
  <p>Gaoling School of<br />Artificial Intelligence<br /><span>Renmin University of China</span></p>
  <p class="profile-location">Beijing, China</p>
  <div class="profile-links">
    <a href="mailto:{{ site.author.email }}" aria-label="Email"><i class="fa-solid fa-envelope" aria-hidden="true"></i></a>
    <a href="https://github.com/{{ site.author.github }}" aria-label="GitHub"><i class="fa-brands fa-github" aria-hidden="true"></i></a>
    <a href="{{ site.author.googlescholar }}" aria-label="Google Scholar"><i class="ai ai-google-scholar" aria-hidden="true"></i></a>
    <a href="https://x.com/Schrieffer875" aria-label="X"><i class="fa-brands fa-x-twitter" aria-hidden="true"></i></a>
  </div>
</aside>

Hi, I'm **Tiancheng Xing**, also known as **Yuu**.

I'm an undergraduate at the **Gaoling School of Artificial Intelligence, Renmin University of China**. My research interests lie in **multimodal reasoning**, **efficient multimodal models**, and **video generation**.

I am particularly interested in understanding and reducing computational redundancy in multimodal models through KV cache compression, token pruning, sparse attention, and adaptive caching.

I also have experience in **visual SLAM and robotic perception**, which has shaped my interest in building efficient, robust systems that can perceive, reason about, and interact with the real world.

<p class="intro-links"><a href="{{ '/publications/' | relative_url }}">Publications <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a href="{{ '/cv/' | relative_url }}">Curriculum vitae <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a href="mailto:{{ site.author.email }}">Email <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></p>
</div>

## Research interests

- **Multimodal reasoning** — Vision-language models, long-horizon reasoning, and visual information flow.
- **Efficient multimodal inference** — KV cache compression, token pruning, sparse attention, and adaptive caching.
- **Video generation** — Diffusion models, flow matching, video DiTs, and feature reuse.
- **Embodied intelligence** — Vision-language-action models, world models, and efficient perception.

<div class="section-heading"><h2>Selected publication</h2><a href="{{ '/publications/' | relative_url }}">All publications <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></div>

{% for paper in site.data.research %}
  {% include research-publication.html paper=paper %}
{% endfor %}

## Selected projects

<div class="project-list">
{% for project in site.data.projects %}
  <article class="project-entry">
    <h3><a href="{{ project.url }}">{{ project.name }} <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></h3>
    <p class="project-area">{{ project.area }}</p>
    <p>{{ project.description }}</p>
  </article>
{% endfor %}
</div>

## Notes & coursework

I also keep course notes and reports on computer systems and artificial intelligence, including LinkLab, AttackLab, CacheLab, BombLab, and DataLab. [Browse my articles and notes →]({{ '/articles/' | relative_url }})

<p class="home-note">“Programming is the art of adding bugs to an empty text file.”</p>
