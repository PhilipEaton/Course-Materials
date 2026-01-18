---
layout: default
title: Mathematical Methods – Homeworks
course_home: /courses/math-methods/
---

# Homework Sets sdkjfndkjfn

<ul>
{% assign course_prefix = page.course_home | append: "homework/" %}

{% assign items = site.pages
  | where_exp: "p", "p.url contains course_prefix"
  | where_exp: "p", "p.nav_section == 'homework'"
  | sort: "nav_order" %}

{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>

<p><strong>DEBUG:</strong> Homework pages detected:</p>
<ul>
{% for p in site.pages %}
  {% if p.nav_section == 'homework' %}
    <li>{{ p.url }} — {{ p.title }}</li>
  {% endif %}
{% endfor %}
</ul>

