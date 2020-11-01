---
layout: post
title:  "Laid off, now what?"
date:   2020-09-19 16:20:13 -0700
---

As an immigrant on an H1B, you have exactly 60 days to find a new job when you are laid-off. This is a very short window of time to explore and land any job, let alone a job that matches your skills and interests. I found myself in this situation along with many others when Uber announced [layoffs](https://www.theverge.com/2020/5/18/21262337/uber-layoff-3000-employees-covid-19-coronavirus) earlier this year. The following is a recollection of some things that worked well for me during my eventually successful job hunt.

- [Always be Prepping](#always-be-prepping)
- [Reaching Out](#reach-out-to-everyone)
- [Interview Preparation](#interview-preparation)
- [Closing Thoughts](#closing-thoughts)

### Always be prepping

Coding interviews are hard to crack if you haven't been prepping, so I was lucky that I had been spending roughly 3-4 hours every week on [leetcode](http://leetcode.com), from about 2 months before the layoff rumours broke. I was lucky that
- I knew that I wanted to change jobs in any case and
- Rumours of layoffs broke approximately a month before the actual layoffs happened, giving me more lead time to prepare and send out emails to recruiters and friends.

Whenever there's an economic downturn, I think it's critical to be acutely aware of what's happening at the company and start preparing for job interviews right away.

### Reach out to everyone

One of the hardest things to do when you are laid off is to write to your friends and family seeking help. But if there's ever a time to swallow your pride, then this is it. I reached out to everyone I knew, and told them plainly about my situation, and asked to be recommended to specific roles at their companies, or to tell their friends who may be hiring. I am extremely grateful to the help I got from my network, and the kind messages that I received. So many friends wrote to make sure I was okay, and kept checking in throughout the interview process, and they all have my immense gratitude. 

It is tempting to just apply on the careers page when you find relevant roles at a company instead of spending time on finding connections and reaching out to them, but in my experience, it was very much worth it. Response times from recruiters was roughly 1-2 days when I was referred by an employee, whereas applying on the careers page was a hit or miss. One BigCo. took 40 days to respond, while some smaller companies were much quicker (3-4 days).

#### The Process

In terms of companies, cast a wide net because you absolutely need _a_ job before a deadline. The steps are the obvious ones:

1. Make a list of companies
2. For each company, compile a list of open job profiles that are relevant.
3. Email/Text a connection at the company, or apply on the careers page if all else fails.

I think I reached out to an initial list of 10 companies or so on the day news of the layoffs broke. This worked well because there's at least a week's time before you speak to a hiring manager or interviewer from when you reach out, so there's ample time to prepare.

What companies to reach out to? In my case, it was the usual suspects (FAANG), and then some domain specific ones such as autonomous vehicle companies. The two most common roles that I applied to were:

- **Machine Learning Engineer** - This is a hybrid role with ML + Software Engineering skills needed, and job roles usually talk about some specific domain such as recommendation systems, or in the case of autonomous vehicles, things such as perception or object detection. I typically looked for some mention of Computer Vision, NLP and deep learning.
- **Machine Learning Infra Engineer** - This role tends to be more on the software systems side, and deals with the infra for training and serving ML models for production workloads.

### Interview Preparation

- [Hiring Manager Screen](#an-initial-screen-with-the-hiring-manager)
- [Coding Interviews](#coding-interviews-phone--onsite)
- [Machine Learning Interviews](#machine-learning-interviews)
- [Behavioral Interviews](#behavioral-interviews)

Interviewing for ML specific roles typically involves a few different kinds of interviews, each of which needs specific preparation. I'm outlining the most common ones I saw below:

### An initial screen with the hiring manager

Companies that do general interviews (Google / Facebook) don't have this step, but most others do. I personally like this, because it means that you are interviewing for a specific position in a specific team, and there's a high level of engagement from the beginning. Most of these calls were about getting to know me, and making sure I have relevant work experience, while some of them also were rapid fire technical questions. The latter ones were rare, and I encountered them when the manager wasn't certain that I was the right person for the job. In my experience, the introduction is the most important part of this interview (**Tell me about yourself**), and it helps to have prepared intros for each type of role that you are applying to. The idea is to tailor your story to highlight aspects of your work experience that are relevant to the job role. The next most important question is "**What would you like to do in your next role?**". Again, it helps immensely to be prepared to answer this question, and ideally, in a way so that there's reasonable overlap between your answer and what the role offers. Being able to answer this question also provides clarity to the job search process. For example, a consistent theme for me was to be (a) in an impactful / critical role for the company and (b) continue to work with the latest in ML.

Writing and rehearsing your stories often seems unimportant when compared to more tangible preparation steps such as spending time on leetcode, but I believe that it was critical, because it sets the tone and gives you confidence that you have done this in the past, and done it well, and there's no reason for the interviewer to doubt your abilities.

### Coding Interviews (Phone / Onsite)

These are the standard leetcode style coding interviews, done using coderpad, or some similar service. The template for these is consistent across all companies, and involves 1 or 2 coding questions (or 1 question with follow-ups) that you are expected to implement and test. Some tips that were helpful for me preparation:

1. **Get a premium subscription with leetcode** - It is nice to be able to filter by companies and have access to the entire question bank, and it is good karma. The service is valuable and the creators should be compensated.
2. **Simulate the interview setting as much as possible** - For example, I would set aside a 3 hour block of time for leetcode, shut myself in a room, and do 4 questions, 45 minutes each. If you are unable to solve a question in 45 minutes, you still move on to the next one. No extensions or looking at the solution. Think of it like moving on to the next interviewer. After the 3 hour session is done, go back to the questions as needed, either to look at solutions or to understand them better. A question is `Done` when your solution passes all the tests on leetcode and is `Accepted`.
3. **Talk out loud** - This is big. Again, assuming that you are in an actual interview, talk out loud about the process you are using during these practice sessions. Talking out loud helps massively because you are forced to put your current train of thought into words, and it is often evident when a solution isn't justifiable.
4. **How to pick questions?** - I filtered for questions that were tagged `Hard`, and then picked at random. No filter for company, or problem type. I went from doing all Mediums to a mix of Mediums and Hard to all Hards over a span of 4-5 weeks.
5. **How many questions to do?** - In the first 2-3 weeks of my prep, I was doing 4 questions on one of the weekend days, and once I had more time post the lay-off news, it was 4 questions every 3 days or so. Overall, my stats look so:

![Leetcode stats](/assets/images/leetcode_xp.png)

### Machine Learning Interviews

These typically come in two flavors:

#### Concepts / Basics

These are kind of like rapid fire questions where the interviewer will quiz you about ML basics. Some questions that I recall right now, to give a flavor of things:

```
- What are some unsupervised learning methods?
- What is underfitting / overfitting?
- What is batch normalization? What's the motivation behind it?
- What is dropout? 
- What optimizers have you used? And typically some follow-up like, why does momentum make sense?
- What are some object detection techniques / papers that you are familiar with? (Computer Vision specific)
- What are decision trees? 
- How does logistic regression work?
- How do you train a linear regression model?
- What are some loss functions that you are familiar with?
- Why does Cross Entropy loss make sense?
- What are residual networks?
```

These are usually follow up questions where the interviewer will try to dig deeper into these concepts, often picking on some portion of the initial answer.

I did a lot of reading, and then some writing with a pen and paper for this part of the interview. If I am already somewhat/fairly familiar with a topic, like say, object detection, then my process was:
1. **Write** from memory a summary of what I remember about the topic
2. Note down **questions** for the parts that I am not clear about
3. **Read** about the topic, and fill in whatever I missed on first go.

If I don't remember much about a topic at all, like say, multi-armed bandits, then I would do step (3) first, and then do steps (1) and (2) a few days later, and eventually repeating step (3) as needed.

It helps to start with a list of topics that you want do this for. This list will grow as you remember more topics or expand the list of companies you are interviewing at. For reference, the list of topics I looked at is [here](/assets/files/index_card.pdf), and a sample of the handwritten notes I made is here, for [recommendation systems](/assets/files/rec_sys.pdf).


#### ML System Design

This is my favorite interview, and corresponds neatly to skills used day to day as a ML practitioner. These are typically open ended interviews where the candidate is expected to design a product with some ML at its core. For example, things like:

```
- Let's build a model that ranks photos in your photo library based on quality.
- How would you build a model that identifies pedestrians from drone imagery?
- Let's build a model that can does face detection for a user's photo library.
- How do you build a model that automatically picks out thumbnails from videos?
- Building some sort of recommendation system.
```

The exact nature of the question and the technologies involved, like Computer Vision or NLP or recommendation systems, depends on the company and the team that you are interviewing for. People generally tend to ask questions on systems that they work on day to day.

As with any system design question, it's useful to draw out the major components first, and then go into the details. Something like this (thanks to my wife for helping with this):

![ML System Design](/assets/images/ml_system_design.svg)

Once you have the major components, you need to cover minimally:
- **Problem Formulation** - How do you formulate the problem? What are the inputs / outputs to the model you are considering?
- **Data** - How do you collect the data needed to train this model? User actions, expert annotations...
- **Model Training** - Specifics on model training. What kind of model, loss functions, evaulation metrics and such
- **Model Serving** - How will the model be deployed? Is it something on device? How many QPS do we need to support? Or are the model predictions computed in a batch offline? Is there versioning involved? Questions of large scale system design are talked about here.
- **System Evaluation** - What metrics would need to be collected? How do we keep the model up to date as time goes on?

The best way to prepare for these interviews is to go through mock practice interview sessions with a friend/co-worker/spouse.

### Behavioral Interviews

This is a more involved or extended version of the initial phone screen with the hiring manager. This is typically where you are expected to talk about how your past experiences have shaped your thought processes and made you a better person (hopefully). The exact format of this interview is variable, but each company typically has a list of core values or competencies they look for, like for example, communication. They have a few different questions for each competency. For example, something like "tell me about a time when you had to convince your team members about a design choice you made".  I prepared for these interviews by:

1. **List Successes** - Write down a list of successful projects and situations from you past that you think you had handled well. Go exhaustive - and try to get to 5-10 situations.
2. **Story Writing** - For each of these, write how you want the story to be told. This is critical, because you'll likely not do the situation any justice if you are talking about it for the first time during the interview. You have to remember that the interviewer knows absolutely nothing about your situation, and hence spending time on setting up context is important. You should then clearly articulate what the challenges were, and how you handled it, and your takeaways. This process gets easier if you write about successful projects when the memory is fresh in your mind. You'll always need to recap your success stories, for the next job or to get a raise. So spending a few hours at the end of a successful project to write a retrospective is a huge force multiplier, and you'll come back to refer to this document often.
3. **Map to Competencies** - Before you interview at a company, look at the values / competencies they look for, and map each of them to some story from step (1). This will be a many-to-many map, so you'll have backups, and you can pick a story during the interview to optimize for impact and diversity of stories.
4. **Rehearse** - Do practice interviews with a friend/co-worker/spouse and rehearse your stories.

During the interview, always take some time to think about how you would modify your story so that it answers the question asked by the interviewer. Don't jump into story-telling. Collect your thoughts first. As you tell your story, make it easy for the interviewer to make their notes by mentioning proxy terms for the skill they are currently testing for.


### Closing Thoughts

Finally, I made some private notes on how I ended up in a situation where I got laid off, and some thoughts on avoiding similar pitfalls in the future are noted here:

- Always ask: Is the work that I am going to be doing in this team critical to the product/business? Note that in the case of larger companies, you should try to understand if the product or business that the team supports is large enough. The hiring manager should be able to clearly and succintly articulate the team's mission. There should be well defined examples of past projects and future milestones that relate to company priorities. For ongoing/future projects, how would stakeholders adopt if this team were to stop existing? 
- If you are joining a research team, the company should be profitable enough and have enough cash to support research activities for the long term, or if not, the research being done has to be critical to the company's mission.
- When you decide to start interviewing for jobs, **always always** talk to multiple companies - it is definitely painful to manage and prepare for multiple interviews, but you need to be able to walk away from a sub-optimal offer, and you can only do that when there are multiple offers or promising processes in the pipeline. 

[Discussion on HN](https://news.ycombinator.com/item?id=24534685)
