---
layout: default
title: Machine Learning Foundations – Examples
course_home: /courses/machine-learning-foundations/
---

# Interesting Examples

<ul>
{% assign course_prefix = page.course_home | append: "interesting-examples/" %}

{% assign items = site.pages
  | where_exp: "p", "p.url contains course_prefix"
  | where_exp: "p", "p.nav_section == 'interesting-examples'"
  | sort: "nav_order" %}

{% for p in items %}
  <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
{% endfor %}
</ul>