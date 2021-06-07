#April 27, 2021

library(ggplot2)
library(hrbrthemes)


df2 <- data.frame(Group=rep(c('Cortex', "Striatum", "Cortex/Striatum Overlap"), each=1),
                Fold_Enrichment=c(2.72, 3.3, 3.5))


df2$Group <- factor(df2$Group, levels = df2$Group)


c4 = c("azure3", "skyblue", "navyblue")

#p <- ggplot(df, aes(x = reorder(f.name, -age), y = age))

p <- ggplot(df2, aes(x = Group, y = Fold_Enrichment))

p<- p + expand_limits( y = 0)

p <- p + scale_y_continuous(expand = c(0, 0), limits = c(0, 4))



p <- p + geom_bar(stat="identity",fill=c4)+
labs(y= "Fold Enrichment with Predicted", x= "Synapse Proteomics")+
  theme_bw()+
    theme(text = element_text(size=20))+
    theme(panel.border = element_rect(colour = "black", fill=NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
    


p+coord_flip()

#April 29, 2021

library(ggplot2)
library(hrbrthemes)

c4 = c("azure3", "navyblue")

df <- read.csv('/Users/karenmei/Documents/SynSig_August2020/revisions/SynSig_Updated/analyze_synsig/adult_mass_spec.csv')
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

        
