
#April 29, 2021

library(ggplot2)
library(hrbrthemes)

#c4 = c("bisque", "orange")

df <- read.csv('/Users/karenmei/Documents/SynSig_August2020/revisions/SynSig_Updated/analyze_synsig/fetal_mass_spec.csv')
ggplot(data=df, aes(x=tissues, y=enrichment, fill=age))+
geom_bar(stat="identity", position=position_dodge())

p <- ggplot(data=df, aes(x=tissues, y=enrichment, fill=age)) +
geom_bar(stat="identity", position=position_dodge())+
scale_fill_brewer(palette='Reds', name="SynSig Genes", labels = c("All", "New"))+
  theme_minimal()
p<- p + expand_limits( y = 0)
p <- p + scale_y_continuous(expand = c(0, 0), limits = c(0, 4))

p+labs(x="Proteomics Screens", y="Fold Enrichment")+
theme_bw()+
theme(text = element_text(size=20))+
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
    theme(legend.position = c(0.2, 0.9))


