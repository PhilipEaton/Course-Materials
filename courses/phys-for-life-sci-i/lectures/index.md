---
layout: default
title: Physics for Life Sciences I – Lecture Notes
course_home: /courses/phys-for-life-sci-i/
---

# Lecture Notes

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