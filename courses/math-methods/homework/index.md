---
layout: default
title: Mathematical Methods – Homeworks
course_home: Course-Materials/courses/math-methods/
---

# Homework Sets

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