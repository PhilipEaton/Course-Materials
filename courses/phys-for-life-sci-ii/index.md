---
layout: default
title: PHYS 2120 – Lectures
course_home: /courses/phys-for-life-sci-ii/
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