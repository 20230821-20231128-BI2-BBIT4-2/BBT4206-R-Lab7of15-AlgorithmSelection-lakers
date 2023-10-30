Business Intelligence Lab Submission Markdown
================
<Lakers>
\<30-10-2023\>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [STEP 1. Install and Load the Required Packages
  —-](#step-1-install-and-load-the-required-packages--)
  - [arules —-](#arules--)
  - [arulesViz —-](#arulesviz--)
  - [tidyverse —-](#tidyverse--)
  - [readxl —-](#readxl--)
  - [knitr —-](#knitr--)
  - [ggplot2 —-](#ggplot2--)
  - [lubridate —-](#lubridate--)
  - [plyr —-](#plyr--)
  - [dplyr —-](#dplyr--)
  - [naniar —-](#naniar--)
  - [RColorBrewer —-](#rcolorbrewer--)

# Student Details

|                                                                                                                                                                                                                                   |                                                              |     |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** \| \| \| 1. 134780 - C - Trevor Okinda \| \| \| \| 2. 132840 - C - Sheila Wangui \| \| \| \| 3. 131749 - C - Teresia Nungari \| \| \| 4. 135203 - C - Tom Arnold \| \| \| \| \| |                                                              |     |
| **GitHub Classroom Group Name**                                                                                                                                                                                                   | Lakers                                                       |     |
| **Course Code**                                                                                                                                                                                                                   | BBT4206                                                      |     |
| **Course Name**                                                                                                                                                                                                                   | Business Intelligence II                                     |     |
| **Program**                                                                                                                                                                                                                       | Bachelor of Business Information Technology                  |     |
| **Semester Duration**                                                                                                                                                                                                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023 |     |

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

# STEP 1. Install and Load the Required Packages —-

## arules —-

if (require(“arules”)) { require(“arules”) } else {
install.packages(“arules”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## arulesViz —-

if (require(“arulesViz”)) { require(“arulesViz”) } else {
install.packages(“arulesViz”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## tidyverse —-

if (require(“tidyverse”)) { require(“tidyverse”) } else {
install.packages(“tidyverse”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## readxl —-

if (require(“readxl”)) { require(“readxl”) } else {
install.packages(“readxl”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## knitr —-

if (require(“knitr”)) { require(“knitr”) } else {
install.packages(“knitr”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## ggplot2 —-

if (require(“ggplot2”)) { require(“ggplot2”) } else {
install.packages(“ggplot2”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## lubridate —-

if (require(“lubridate”)) { require(“lubridate”) } else {
install.packages(“lubridate”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## plyr —-

if (require(“plyr”)) { require(“plyr”) } else { install.packages(“plyr”,
dependencies = TRUE, repos = “<https://cloud.r-project.org>”) }

## dplyr —-

if (require(“dplyr”)) { require(“dplyr”) } else {
install.packages(“dplyr”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## naniar —-

if (require(“naniar”)) { require(“naniar”) } else {
install.packages(“naniar”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }

## RColorBrewer —-

if (require(“RColorBrewer”)) { require(“RColorBrewer”) } else {
install.packages(“RColorBrewer”, dependencies = TRUE, repos =
“<https://cloud.r-project.org>”) }
