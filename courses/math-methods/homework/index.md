---
layout: default
title: Mathematical Methods – Homeworks
course_home: courses/math-methods/
---

# Homework Sets

<ul>
{% assign items = site.pages
  | where_exp: "p", "p.course_home == page.course_home"
  | where_exp: "p", "p.nav_section == 'homework'"
  | sort: "nav_order" %}

{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>
