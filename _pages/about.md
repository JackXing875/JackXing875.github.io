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
  <img src="{{ '/images/profile.png' | relative_url }}" alt="Yuu's profile avatar" width="280" height="280" fetchpriority="high" />
  <p>Gaoling School of<br />Artificial Intelligence<br /><span>Renmin University of China</span></p>
  <p class="profile-location">Beijing, China</p>
  <div class="profile-links">
    <a href="mailto:{{ site.author.email }}" aria-label="Email"><i class="fa-solid fa-envelope" aria-hidden="true"></i></a>
    <a href="https://github.com/{{ site.author.github }}" aria-label="GitHub"><i class="fa-brands fa-github" aria-hidden="true"></i></a>
    <a href="{{ site.author.googlescholar }}" aria-label="Google Scholar"><i class="ai ai-google-scholar" aria-hidden="true"></i></a>
    <a href="https://x.com/Schrieffer875" aria-label="X"><i class="fa-brands fa-x-twitter" aria-hidden="true"></i></a>
  </div>
</aside>

I'm an undergraduate at the **Gaoling School of Artificial Intelligence, Renmin University of China**. My research interests lie in **multimodal reasoning**, **efficient multimodal models**, and **video generation**.

I am particularly interested in understanding and reducing computational redundancy in multimodal models through KV cache compression, token pruning, sparse attention, and adaptive caching.

I also have experience in **visual SLAM and robotic perception**, which has shaped my interest in building efficient, robust systems that can perceive, reason about, and interact with the real world.

<p class="intro-links"><a href="{{ '/publications/' | relative_url }}">Publications <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a href="{{ '/cv/' | relative_url }}">Curriculum vitae <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a href="mailto:{{ site.author.email }}">Email <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></p>
</div>

<section class="home-section" aria-labelledby="research-interests">
  <div class="section-heading"><h2 id="research-interests">Research interests</h2></div>
  <div class="interest-list">
    {% for interest in site.data.interests %}
      <div class="interest-entry">
        <h3>{{ interest.title }}</h3>
        <p>{{ interest.description }}</p>
      </div>
    {% endfor %}
  </div>
</section>

<section class="home-section" aria-labelledby="selected-publication">
<div class="section-heading"><h2 id="selected-publication">Selected publication</h2><a href="{{ '/publications/' | relative_url }}">All publications <span aria-hidden="true">&rarr;</span></a></div>

{% for paper in site.data.research %}
  {% include research-publication.html paper=paper %}
{% endfor %}

</section>

<section class="home-section" aria-labelledby="selected-projects">
<div class="section-heading"><h2 id="selected-projects">Selected projects</h2><a href="https://github.com/{{ site.author.github }}">GitHub <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></div>

<div class="project-list">
{% for project in site.data.projects %}
  <article class="project-entry">
    <h3><a href="{{ project.url }}">{{ project.name }} <i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></h3>
    <p class="project-area">{{ project.area }}</p>
    <p>{{ project.description }}</p>
    <p class="project-tools">{{ project.tools }}</p>
  </article>
{% endfor %}
</div>

</section>

<section class="home-section" aria-labelledby="recent-writing">
  <div class="section-heading"><h2 id="recent-writing">Recent writing</h2><a href="{{ '/articles/' | relative_url }}">All articles <span aria-hidden="true">&rarr;</span></a></div>
  <p class="section-intro">Notes and experiments in artificial intelligence and computer systems.</p>
  <div class="article-list">
    {% assign recent_notes = site.publications | sort: 'date' | reverse %}
    {% for note in recent_notes limit:3 %}
      {% include note-entry.html note=note compact=true %}
    {% endfor %}
  </div>
</section>
