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

