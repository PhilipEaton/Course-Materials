---
layout: default
title: Machine Learning Foundations – Lectures
course_home: /courses/machine-learning-foundations/
---

# Lectures

<ul>
{% assign course_prefix = page.course_home | append: "lectures/" %}

{% assign items = site.pages
  | where_exp: "p", "p.url contains course_prefix"
  | where_exp: "p", "p.nav_section == 'lectures'"
  | sort: "nav_order" %}

{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>
