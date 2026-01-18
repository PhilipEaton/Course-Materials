---
layout: default
title: Physics for Life Sciences I – Practice
course_home: /courses/phys-for-life-sci-i/
---

# Homework Sets

<ul>
{% assign course_prefix = page.course_home | append: "practice/" %}

{% assign items = site.pages
  | where_exp: "p", "p.url contains course_prefix"
  | where_exp: "p", "p.nav_section == 'practice'"
  | sort: "nav_order" %}

{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>