---
slug: data-science/visualisation/how-to-do-sentiment-analysis-on-your-favourite-book-with-r
title: How To Do Sentiment Analysis On Your Favourite Book With R
tags: [Data Science, R, Sentiment Analysis, Text Analytics Visualisation]
---

I love dissecting and analysing my favourite books by reading them again and again, discussing them with my like-minded friends, getting to know the authors and reading other people’s writings about them. <!--truncate--> My obsession with books lead me to thinking, how can I visualise them in interesting ways? Making Word Cloud is an easy and fun way to visualise your favourite book as in the previous post. In this post, we will go down deeper into the world of text analytics by using sentiment analysis. I will show you how to split the text by sentence, conduct sentence-wise sentiment analysis and create an interactive plot that shows how sentiment changes as the story progresses.

There are many different ways to do sentiment analysis. The easiest way is to split the sentence by word and score each word by looking up word sentiment dictionary. The tidytext packages in R has a build in function to do a basic sentiment analysis. The package documentation from CRAN shows sentiment analysis on Jane Austin text. This one with Harry Potter is also fun to read. For more advanced Natural Language Processing (NLP), you can use Stanford CoreNLP, which is very powerful, but cumbersome to use and can be slow at a time.

In this post, I am using sentimentr. The package is fast and super easy to use. It does sentence-wise sentiment analysis by using dictionary lookup and valence shifter. With valence shifter, it updates the normal word sentiment dictionary look up score with the score by looking at polarised word (such as really or hardly). It reflects the sentiment more accurately than just scoring individual words. It also has the get_sentense function to splice the text into sentences by using regular expressions.

The `sentimentr` package really makes sentiment analysis pop and accessible. I really love using it.

Sentence analysis on literature is hard because of slang, unconventional style and expressions, unambiguous sentence separation, sarcasms, jokes and so on. Despite the outcome is not perfect, you’ll have a lot of fun!

As the continuation from the previous post, I am using one of my favourite book, On The Road by the greatest Beat hero Jack Kerouac.

Ok, let’s code!

### Summary Steps

1. Download a pdf file and convert it into text.
2. Create a data frame with row per sentence and do clean up.
3. Further split the sentence by get_sentence() and create a new data frame
4. Get sentiment per sentence.
5. Plot the sentiment by `plotly`.
6. Check the results and have fun!

### Steps

(1) Get the pdf file of On The Road from `freeditorial.com` and use `pdftools` to convert it to text. This step is the same as in the previous post.

```R
library(pdftools)
download.file("https://freeditorial.com/en/books/on-the-road/downloadbookepub/pdf",
"/tmp/on_the_road.pdf", mode = "wb")
txt <- pdf_text('/tmp/on_the_road.pdf')
```

(2) Create the first data frame with row by sentence. For sentiment analysis, cleaning up the text has to be a little bit more diligent than just making a word cloud. You just need to look at your text file to determine what is needed. The regular expression is the way to go to clean up the text data. Use online resource for detailed examples as well as a quick reference. Make sure to test your regex with a short example!

```R
lines <- ''
for (i in 2:length(txt)) {
tmp <- gsub('\r\n', ' ', txt[i]) # \r\n for Windows, \n for Linux
lines <- paste(lines, tmp, sep=' ')
}

vec <- strsplit(lines, '\\.')
df <- data.frame(vec)

df <- as.data.frame(df[-c(nrow(df), nrow(df)-1), ]) # Remove Last 2 lines
colnames(df)[1] = 'line' # Rename Columns

df$line <- gsub("«", "", gsub("»", "", df$line))
df$line <- gsub('^\\s+PART\\s+[A-Z]+\\s+', '', df$line)
df$line <- as.character(trimws(df$line, 'both'))
df$line <- gsub('^[1-5]\\s{2,}', '',  df$line)
df$line <- gsub('- -, ', '',df$line)
```

(3) Further split the sentences by the power of get_sentences function. This step is necessary to get a sentiment score per sentence according to the `sentimentr` package. The sentiment function in `sentimentr` may returns more than one sentiment score per sentence if you do not do this.

```R
library(sentimentr)
sentence <- c()
for (line in df$line) {
tmp <- get_sentences(line)
for(i in 1:length(tmp[[1]])) {
sentence_tmp <- tmp[[1]][i]
sentence <- c(sentence, sentence_tmp)
}
}

df_sentr <- data.frame(sentence)
df_sentr$sentence <- as.character(df_sentr$sentence)
(4) Once your data frame is ready, do the scoring with sentiment(). Add negative, positive and neutral indicator for visualisation.

sentiment <- sentiment(df_sentr$sentence)

df_sentr$sentiment <- as.numeric(sentiment$sentiment)
df_sentr$pntag <- ifelse(sentiment$sentiment == 0, 'Neutral',
ifelse(sentiment$sentiment > 0, 'Positive',
                                ifelse(sentiment$sentiment < 0, 'Negative', 'NA')))
```

(5) You can plot it with base R plot function. I like using Plotly because it becomes interactive. The x-axis shows the progress of the story and the y-axis shows the sentiment. Positive, negative and neutral sentiments are colour-coded. Once you hover the mouse over the dot, you can see the sentence. The `magrittr` package enables the magic of the pipe (%>%) operation.

- base R plot

```R
plot(df_sentr$sentiment, type='l', pch=3)
```

- plotly- more fun

```R
ax <- list(
title = "Sentence",
zeroline = FALSE,
showline = FALSE,
showticklabels = FALSE
)

library(plotly)
library(magrittr)
plot_ly(data = df_sentr, x = ~sentence, y = ~sentiment, color = ~pntag,
type = 'scatter', mode = 'markers') %>% layout(xaxis = ax)
plot_ly(data = df_sentr, y = ~sentiment, color = ~pntag,
type = 'scatter', mode = 'markers') %>% layout(xaxis = ax)
```

Output Example

![Output](./img/sentiment-graph.webp)

(6) Let’s check what is the most positive and the most negative and calculate the average sentiment score. I also created csv files with sentiment score above 1 and below -1.

- Check max and min sentiments

```R
df_sentr$sentence[which.max(df_sentr$sentiment)]
df_sentr$sentence[which.min(df_sentr$sentiment)]
```

- Get average

```R
mean(df_sentr$sentiment)
```

- Check positive and negative sentences

```R
check_pos <- subset(df_sentr, df_sentr$sentiment >= 1.0)
check_pos <- check_pos[order(check_pos$sentiment, decreasing = T),]
write.csv(check_pos, "positive.csv")
check_neg <- subset(df_sentr, df_sentr$sentiment <= -1.0)
check_neg <- check_neg[order(check_neg$sentiment, decreasing = F),]
write.csv(check_neg, "negative.csv")
```

## Discussion

The average sentiment score was 0.024. All the highs and lows must have cancelled each other out, coming out slightly positive just like life in general.

The most positive moment in the story according to sentimentr is absolutely gold.

- I think Marylou was very, very wise leaving you, Dean, said Galatea

- Yeah, that’s pretty much the best decision! On a more serious note, this is the prime example of the effect of using valence shifter in the sentiment algorithm. It has two positive polarised words (very, very) on a positive word (wise). Very twice really amplified the positive sentiment on this sentence.

- Not sure about the most negative one. But, this is also the fun part, getting misplaced sentiment by an algorithm.

- He showed me rooming houses where he stayed, railroad hotels, poolhalls, diners, sidings where he jumped off the engine for grapes, Chinese restaurants where he ate, park

Looking at the results, I think it performed well overall despite questionable sores here and there. To be fair, On The Road is a difficult one to run sentiment algorithm because it is written in a stream-of-consciousness style with many unique expressions and unclear sentence break points.

Here are the top 6 positive sentences. No.2 is the classic Dean Moriarty haunted by his own madness. No.4 and 5 are questionable. No.6 is absolutely yes, I think.

1. I think Marylou was very, very wise leaving you, Dean, said Galatea
2. Hmm, ah, yes, excellent, splendid, harrumph, egad!
3. It certainly was pleasant, said Hingham, looking away
4. Doll, Dorothy and Roy Johnson, a girl from Buffalo, Wyoming, who was Dorothy’s friend, Stan, Tim Gray, Babe, me, Ed Dunkel, Tom Snark, and several others, thirteen in all
5. There’s one last thing I want to know – But, dear Sal, you’re listening, you’re sitting there, we’ll ask Sal
6. Absolutely, now, yes?

Here are the bottom 6 negative sentences. Yeah, it certainly shows some low moments in the book despite of some questionable entries. No. 5 is another classic Dean Moriarty.

1. He showed me rooming houses where he stayed, railroad hotels, poolhalls, diners, sidings where he jumped off the engine for grapes, Chinese restaurants where he ate, park benches where he met girls, and certain places where he’d done nothing but sit and wait around
2. There was something paralyzed about his movements, and he did nothing about leaving the doorway, but just stood in it, muttering, Stan, and Don’t go, and looking after us anxiously as we rounded the comer
3. Then the third day began having a terrible series of waking nightmares, and the were so absolutely horrible and grisly and green that I lay there doubled up with my hands around my knees, saying, ‘Oh, oh, oh, ah, oh
4. Offisah, I heard Dean say in the most unctuous and ridiculous tones, offisah, I was only buttoning my flah
5. And the one to my left here, older, more sure of himself but sad
6. But this foolish gang was bending onward

Wow, what a fun I had!

Now, you are ready for sentiment analysis on your favourite book!
