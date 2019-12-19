# Time to Build Random Forest Graphs
times <- read.csv("time_methods.csv")

library(ggplot2)

ggplot(times, mapping=aes(x = method, y=time, fill=as.factor(rows))) + geom_boxplot() +
  labs(fill="Number of Rows", y="Time (s)") + theme_bw()


ggplot(times, mapping=aes(x = method, y=time, fill=as.factor(columns))) + geom_boxplot() +
  labs(fill="Number of Columns", y="Time (s)") + theme_bw()

ggplot(filter(times, method=="CreateDT"), mapping=aes(x=))

dtonly <- read.csv("time_dt.csv")
dtagg <- aggregate(.~rows+columns, dtonly, mean)
ggplot(dtagg, aes(x=rows, y=time, color=as.factor(columns))) + 
  geom_point() + theme_bw() + labs(color="Columns", x = "Rows", y="Average Time to Create Tree (s)") +
  geom_line(data = subset(dtagg, columns==2)) +
  geom_line(data=subset(dtagg, columns==4)) +
  geom_line(data=subset(dtagg, columns==6)) + 
  geom_line(data=subset(dtagg, columns==8)) +
  geom_line(data=subset(dtagg, columns==10)) +
  geom_line(data=subset(dtagg, columns==15))



ggplot(dtagg, aes(x=columns, y=time, color=as.factor(rows))) + 
  geom_point() + theme_bw() + labs(color="Rows", x = "Columns", y="Average Time to Create Tree (s)") +
  geom_line(data = subset(dtagg, rows==25)) +
  geom_line(data=subset(dtagg, rows==50)) +
  geom_line(data=subset(dtagg, rows==75)) + 
  geom_line(data=subset(dtagg, rows==100)) +
  geom_line(data=subset(dtagg, rows==200)) 