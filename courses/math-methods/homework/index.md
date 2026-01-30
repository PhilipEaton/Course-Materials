---
layout: default
title: Mathematical Methods – Homeworks
course_home: /courses/math-methods/
---

# Homework Sets

{% assign course_prefix = page.course_home | append: "homework/" %}

{% assign items = site.pages
  | where_exp: "p", "p.url contains course_prefix"
  | where_exp: "p", "p.nav_section == 'homework'"
  | sort: "nav_order" %}

{% assign groups = items | group_by: "hw_type" %}

{% for g in groups %}
  <h2>
    {% if g.name == "short" %}Short Homework{% elsif g.name == "long" %}Long Homework{% else %}{{ g.name | capitalize }}{% endif %}
  </h2>

  <ul>
    {% for p in g.items %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
