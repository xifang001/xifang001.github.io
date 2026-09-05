---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
lastmod: {{ .Date }}
slug: "{{ .Name }}"
summary: "A concise summary of the robotics note."
description: "A slightly fuller description for previews, metadata, and the blog index."
tags:
  - Robotics
  - Topic One
  - Build Notes
categories:
  - Robotics
series:
  - Robotics Notes
authors:
  - Xi Fang
toc: false
draft: true
part_label: ""
part_summary: ""
---

Use this template for robotics notes, system bring-up, kinematics, manipulation, or deployment writeups.

## Context

What system, task, or technical question led to this post?

## Main Notes

Organize the post into 2-4 focused sections.

### Hardware or Setup

Describe the setup, assumptions, or environment.

### Method or Workflow

Explain the core technical idea, workflow, or build process.

### Debugging or Observations

Capture the useful details, failure cases, or lessons learned.

## Closing Notes

End with the main takeaway, next step, or open question.
