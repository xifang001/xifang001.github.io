---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
lastmod: {{ .Date }}
slug: "{{ .Name }}"
summary: "One-sentence summary for the post."
description: "A slightly fuller description for previews, metadata, and the blog index."
tags:
  - Topic One
  - Topic Two
categories:
  - Research Notes
series:
  - Writing Notes
authors:
  - Xi Fang
toc: false
draft: true
part_label: ""
part_summary: ""
---

Choose one main category for the post:
- Robotics
- Computer Vision
- Machine Learning
- Research Notes
- Life

Suggested tags:
- core topic, e.g. Kinematics / NeRF / Registration / VLM
- format, e.g. Learning Notes / Build Notes / Research Notes
- system or method, if relevant

Start with 1-3 short opening paragraphs:
- what prompted the post
- what the post is trying to explain, document, or think through
- why it is worth writing down now

## Context

State the setup, question, or motivation clearly.

## Main Notes

Organize the post into 2-4 focused subsections. Keep each section on one thread.

### Section One

Explain one idea clearly, then connect it to an example, observation, or implication.

### Section Two

Continue with one clear thread. Prefer shorter paragraphs over dense blocks.

## Closing Notes

End with the main takeaway, next step, or open question.
