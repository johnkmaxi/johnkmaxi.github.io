---
layout: post
title: "Software Development Processes for Data Scientists"
date: 2018-06-30
---
# Software Development Processes for Data Scientists

Like many data scientitsts, I learned programming because it was a tool I could use to get insights from my data quicker. As I continued working on my graduate work, MOOCs, and my own projects, I continued to pick up better coding practices. These skills got me through all the challenges I needed to solve. However, working as a data scientist for the past year has been a humbling experience in all I don't know about computer science and software engineering. A data scientist is often thought of as a mythical unicorn that knows all of statistics, all of computer science, and all of machine learning. In reality, I think data scientists are just people who love to learn, so when they don't know something, they go read about it or try it out. In that spririt, I took the Software Development Process from Udacity to get up to speed on these concepts. In this post, I will present an overview of some of the major concepts and types of software development processes.

By the end of the post you should have an understanding of five common devlopment processes and the pros and cons of each type as they relate to data scientists.

### Waterfall
In the waterfall design method, there is a clear progression and discrete steps in the development process. Generally, the software is conceptualized and then there is a requirements analysis to determine everythin that will be needed for the project. Next is the architectural design phase in which the software stucture is decided upon. After architechtural design comes the detailed design phase (thinking writing psuedocode for all functions and classes). In the fifth phase, the coding and the debugging occur. Finally, the system undergoes rigorous testing prior to deployment. The phases occur in a linear order and there is a review at the end of each phase to determine if the team is ready to move on to the next phase. The waterfall development process works well when there is a stable product definition, the project domain is well known, and the technologies involved are well understood. Essentially, the waterfall design works well when dealing with straightforward project with very few unknowns. For example, developing software to trigger a camera to take a picture in response to identifying a particular face. The software needed to trigger a camera and save images is well understood and straightforward. The product team and software engineers are not likely to have a problem identifying the requirements and design of the system. For many data scientists, the waterfall development process is not feasible due to the experimental nature of their work. However, the example above shows how a data scientist could work with a team of engineers on a relatively well known problem of facial recognition to provide a piece of overall product software package.

**Pros:** easier to find errors early on in the process
**Cons:** not flexible, can be difficult to know all requirements ahead of time

### Spiral
The spiral software design model incorporates the linear design of the waterfall model with an iterative development cycle. The four phases of the spiral model are 1) determine objectives, 2) identify and resolve risks, 3) development and test, and 4) plan next iteration. These 4 phases are carried out in order several times over, resulting in a linear iterative process. The spiral model allows for  a good complement between up-front planning and iterative development. Data scientists may find the spiral model attractive because it gives them an opportunity to formalize their plan, consider what may go wrong, develop and then replan the next phase of development. This iterative process is an unspoken core of the scientific method in which hypothesis are made, tested, and refined as the scientiest moves toward and makes new discoveries. 

**Pro:** risk reduction, functionality can be added in later phases, software is produced early in the process, developmers can get early feed back.
**Cons:** risk analysis requires highly specific expertise, model success is highly dependent on risk analysis, more complex than other models

### Evolutionary Prototyping
Evolutionary prototyping software development that recognizes not all parts of the system are understood at the same time. Therefore, only the parts that are understood should be constructed. Evolutionary prototyping plans for developers to add features and functionality at later times under the assumption that those functions could not have been planned for during the requirements phase. Evolutionary development often results in customers using "incomplete" versions of the product. Customers use the early versions and give feedback to the developers regarding current and future functionaly. Developers then add additional functionality in response to user feedback. Evolutionary Prototype is considered less risky of a development process because programmers do no implement features they do not understand, resulting in code and functionality that the developer fully understands. There are four main phases of evolutionary prototyping software development, 1) initial concept, 2) design and implement initial prototype, 3) refine prototype until acceptable, 4) complete and release the prototype. After the prototype is released, developers will be able to make improvements based on user feedback. Data scientists may prefer evolutionary prototyping because it allows them to get implement minimal functionality in order to get user feedback.

**Pros:** works well when all the requirements are not understood, get immediate feedback
**Cons:** don't know how many iterations will be needed, a process of coding and fixing bugs when found can lead to a low quality product

### Rational Unified Process (RUP)
RUP consists of three main processes: 1) Roles, which are a defined set of related skills, compentencies, and responsibilities. 2) Work products - anything resulting from a task including documentation. 3) Tasks - a unit of work assigned to a role. These processes serve as the building blocks for the four phases and 6 processes utlized by the RUP model. The phases are inception, elaboration, construction, and transition. The phases describe how the product begins as a concrete idea with stakeholders agreeing on the scope, schedule, and cost of the product all the way to finished product. Like in the waterfall model, criteria need to be met before moving from one phase to the next. During these four phases, work is divided into six different processes - business modeling, requirements, analysis and design, implementation and deployment.

**Pros:** stresses good documentation; deployment occurs throughout the process, helping to mitigate layer deployment errors; iterative
**Cons:** early integration can cause problems when non-compatible versions are integrated; considered a complicated/complex model

### Agile
The Agile development processes is the most similar to a scientist's workflow - it is highly incremental and iterative. A usually writes a grant grant and develops a plan of how all the work will go and what the outcome of the resarch will be. The scientist then tests the grants hypotheses and analyzes the results. Often, the experiments don't work out right the first time, and the scientist must troubleshoot and tweak the experiment parameters. As the overall project advances, new questions are bound to arise leading the scientist to incorporate additional experiments into his or her research plan. Agile software development follows an analogous process. The development team works with the customers to plan the overall project. The developers then develop small bits of code, testing as they go to make sure everything is working. The test driven development aspect of Agile is also similar to scientist scenario described above. It consists of three stages, 1) writing test cases that fail, 2) writing enough code for the test cases to pass, and 3) refactoring. Unlike other development modes, e.g., waterfall, if new features are required during development, they are incorporated into the project relatively easily.

**Pros:** iterative, flexible, less focus on documentation, short feedback loop
**Cons:** cannot provide detailed future plans, often inefficient for large organizations

## Which Model Should a Data Scientist Use?
It depents (obvioiusly, right?). It depends on how well the requirements are understood, the expected lifetime of the code, the level of risk, schedule constratins, interaction with management/customers, and the expertise of people involved. For a data scientist, you might be working on research-type projects in which the requirements are not understood and you are not even sure if it will work (aka new deep learning application)! That kind of situation calls for a more iterative approach such as Agile or Evolutionary Prototyping. Or you could be working on building a system that is more well known, like building a web app to deploy a model known to work well. In that case, it is easier to know how, when, and what needs to be done.

### Classic Mistakes
Finally, keep these classic software development mistakes involving people, processes, product, and technology.

People: to much emphasis on single person taking on too much of the project, work environment, poor management, adding new people slows the project down because the new people need to be brought up to speed.

Process: scheduling issues such as having an unrealistic schedule, insufficient or abandoning planning

Product: adding more features than are needed, feature creep (adding features that are not necessary), research and development (While is a big portion of a data scientist's job, it is important to not let it get in the way of releasing a product. Use a simpler solution to ge the product out and do R&D in the background for future releases.)

Technology: silver-bullet syndrome (tech cannot solve all problems), switching tools mid-development, no version control

References:

https://airbrake.io/blog/sdlc/rational-unified-process
https://en.wikipedia.org/wiki/Agile_software_development



