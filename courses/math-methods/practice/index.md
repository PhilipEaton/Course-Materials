---
layout: default
title: Mathematical Methods – Practice Days
course_home: /courses/math-methods/
---

# Practice Days

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