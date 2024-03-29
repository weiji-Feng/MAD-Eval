# Comparison of instructions obtained by different sampling strategies

We qualitatively analyzed the differences among the four strategies in instruction selection. <a href='#table1'>Table 1</a> presents the specific content of the top-10 instructions chosen by different strategies in the writing domain. We found that the instructions selected by the KL divergence strategy are almost all related to poetry (9 out of 10), while the cross-entropy strategy favors tasks such as writing papers and stories. The random strategy often introduces varying numbers of overlapping task types. However, our work proposes the MAD competition strategy, which considers instruction diversity, thus minimizing the occurrence of repeated task types in the selected instructions as much as possible, facilitating the exposure of failures of the LLM across a wider range of tasks.

<a id="table1"></a>
![](../figs/instructions.png)
<center><p>Table 1. The differences in the Top-10 instructions chosen by four sampling strategies. </p></center>