#April 27, 2021
library(ggplot2)
library(dplyr)
library(hrbrthemes)

# Load dataset from github
df <- data.frame(Number_of_Consensus_Sources=c(0, 1, 2, 3, 4), Fold_Enrichment=c(0.44647887323943664, 1.5, 2.0652173913043477, 3.0416666666666665, 3.8048780487804876), PValue=c(8.2803424e-12, 11.7334699124, 23.6512999175, 37.0976806758, 53.9971414619))

# Plot
df %>%
  ggplot( aes(x=Number_of_Consensus_Sources, y=Fold_Enrichment, size=PValue)) +
    geom_line( color="grey", size=2) +
    geom_point(shape=21, color="black", fill="darkred", size=10) +
    theme_ipsum() +
    theme_bw()+
    expand_limits(y = 0)+scale_y_continuous(expand = c(0, 0), limits = c(0, 4.5))+

    labs(y= "Fold Enrichment of Novel SynSig Genes", x = "Number of Consensus Sources")+
    theme(text = element_text(size=15))+
    #theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
    
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


#==================
#July 7, 2021
library(ggplot2)
library(dplyr)
library(hrbrthemes)

# Load dataset from github
df <- data.frame(Number_of_Consensus_Sources=c(1, 2, 3, 4), Fold_Enrichment=c(1.4910179640718564, 2.111111111111111, 3.2444444444444445, 4.588235294117647), PValue=c(3.0122809235351785e-12, 1.8084389035305461e-25, 5.530472270965264e-41, 1.2667359701174018e-68))

# Plot
df %>%
  ggplot( aes(x=Number_of_Consensus_Sources, y=Fold_Enrichment, size=PValue)) +
    geom_line( color="grey", size=2) +
    geom_point(shape=21, color="black", fill="darkred", size=10) +
    theme_ipsum() +
    theme_bw()+
    expand_limits(y = 0)+scale_y_continuous(expand = c(0, 0), limits = c(0, 5))+

    labs(y= "Fold Enrichment of Novel SynSig Genes", x = "Number of Consensus Sources")+
    theme(text = element_text(size=15))+
    #theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
    
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggplot(data=df, aes(x=tissues, y=enrichment, fill=age))+
geom_bar(stat="identity", position=position_dodge())

p <- ggplot(data=df, aes(x=tissues, y=enrichment, fill=age)) +
geom_bar(stat="identity", position=position_dodge())+
scale_fill_manual(values=c4, name="SynSig Genes", labels = c("All", "New"))+
  theme_minimal()
p<- p + expand_limits( y = 0)
p <- p + scale_y_continuous(expand = c(0, 0), limits = c(0, 4))

p+labs(x="Proteomics Screens", y="Fold Enrichment")+
theme_bw()+
theme(text = element_text(size=20))+
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
    theme(legend.position = c(0.87, 0.25))

# Load dataset from github
df <- data.frame(Number_of_Consensus_Sources=c(1, 2, 3, 4), Fold_Enrichment=c(1.4910179640718564, 2.111111111111111, 3.2444444444444445, 4.588235294117647), PValue=c(3.0122809235351785e-12, 1.8084389035305461e-25, 5.530472270965264e-41, 1.2667359701174018e-68))

p<-ggplot(data=df, aes(x=Number_of_Consensus_Sources, y=Fold_Enrichment)) +
  geom_bar(stat="identity", fill="darkred")+theme_minimal()


p+labs(x="Proteomics Screens", y="Fold Enrichment")+
theme_bw()+
theme(text = element_text(size=20))+
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+ 
    expand_limits( y = 0)+scale_y_continuous(expand = c(0, 0), limits = c(0, 5))


