# Mastering Large Language Models: Training and Fine-Tuning Large Language Models Workshop (DHS 2024)

This repository contains all the presentations and hands-on code notebooks for the "Training and Fine-Tuning Large Language Models" workshop held at [Analytics Vidhya DataHack Summit 2024](https://www.analyticsvidhya.com/datahacksummit/workshops/mastering-llms-training-fine-tuning-and-best-practices). 
The workshop is divided into several modules, each focusing on different aspects of working with large language models (LLMs) from prompting to fine-tuning

__Workshop instructor:__ [Dipanjan (DJ) Sarkar](https://www.linkedin.com/in/dipanjans/)

<br/>

## Tools and Frameworks Used

<div>
  <img align="left" width="150" src="https://i.imgur.com/Eoxjs0J.png" hspace="10"/>
  <img align="left" width="100" src="https://i.imgur.com/KhQJXz3.png" hspace="10"/>
  <img align="left" width="150" src="https://i.imgur.com/1ZMOoc7.png" hspace="10"/>
</div>
<br clear="left"/><br/>

<div>
  <img align="left" width="300" src="https://i.imgur.com/rsw8Wjb.png" hspace="0"/>
  <img align="left" width="160" src="https://i.imgur.com/fd7fi09.png" hspace="0"/>
</div>
<br clear="left"/>

<br/>

## Workshop Overview

This workshop is divided into 5 modules, the following graphic clearly showcases what is covered in each module.
It is recommended to navigate sequentially through each module to get the best experience. 
Also do go through the presentation decks to learn the conceptual aspects of the topics besides trying out the hands-on python notebooks.

__Note:__ For environment setup each module has its own `requirements.ipynb` notebook with all the necessary libraries needed to be installed. We use [Runpod.io](https://www.runpod.io/) A40 Servers which give us a 48GB VRAM GPU which is enough for these experiments. Make sure you have a separate disk volume of at least 30GB to store the Llama 3 LLM weight files after fine-tuning, else you will get an error.

![](https://i.imgur.com/QJHw6Nh.png))

<br/>

## Modules Overview

### Module 01: Transformers, LLMs, and Generative AI Essentials

<img align="left" width="150" src="https://i.imgur.com/nnrHte5.png"/> <br/> This module serves as the foundational introduction to Transformers, Foundation models, and fine-tuned Large Language Models (LLMs). You'll learn about both simple and contextual embedding models and their applications in real-world scenarios. Practical exercises include building a custom search engine and leveraging Transformer-based LLMs for tasks such as sentiment analysis, Q&A, summarization, and zero-shot classification. A special focus is given to using Microsoft Phi-3 Mini Instruct locally for prompt engineering and comparing performance across models like GPT-4o mini and Meta Llama 3.1 8B.

<br/><br/><br/>

- **[Module_01_Install_Requirements.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Module_01_Install_Requirements.ipynb)**: A notebook to set up the required environment and dependencies for the exercises in this module.
- **[Module_01_LC1_Simple_Word_Embedding_Models_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Solutions/Module_01_LC1_Simple_Word_Embedding_Models_Solutions.ipynb)**: Introduction to simple word embedding models, providing hands-on exercises to understand their workings.
- **[Module_01_LC2_Contextual_Embeddings_and_Semantic_Search_Engines_with_Transformers_Solutions](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Solutions/Module_01_LC2_Contextual_Embeddings_and_Semantic_Search_Engines_with_Transformers_Solutions.ipynb).ipynb**: Explores the use of contextual embeddings and the implementation of semantic search engines using Transformer models.
- **[Module_01_LC3_Real_World_Applications_with_Fine_tuned_Transformers_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Solutions/Module_01_LC3_Real_World_Applications_with_Fine_tuned_Transformers_Solutions.ipynb)**: Demonstrates real-world applications of fine-tuned Transformers in various domains, such as sentiment analysis, Q&A, and summarization.
- **[Module_01_LC4_Prompt_Engineering_with_Local_Open_LLMs_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Solutions/Module_01_LC4_Prompt_Engineering_with_Local_Open_LLMs_Solutions.ipynb)**: Covers techniques for prompt engineering with locally hosted open-source LLMs like Microsoft Phi-3 Mini, including tasks like zero-shot classification and summarization.
- **[Module_01_LC5_BONUS_Comparing_Llama_3_1_vs_GPT_4o_mini_Walkthrough.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-01-Transformers-LLMs-Generative-AI-Essentials/Module_01_LC5_BONUS_Comparing_Llama_3_1_vs_GPT_4o_mini_Walkthrough.ipynb)**: A bonus walkthrough comparing Llama 3.1 and GPT-4o mini models in different tasks using prompt engineering.

---

### Module 02: Pre-training and Fine-tuning LLMs

<img align="left" width="200" src="https://i.imgur.com/ZWqfFcI.png"/> <br/> This module dives deep into the processes of pre-training and fine-tuning LLMs. You'll explore the key steps for building LLMs or SLMs (Small Language Models) from scratch, with hands-on experience in customizing models like GPT-2 through pre-training on unlabeled datasets. 
The module also covers the process of full fine-tuning of pre-trained BERT models for specific tasks like text classification, helping you understand the key workflows of adapting LLMs to specialized applications.

<br/>

- **[Module_02_Install_Requirements.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-02-Pre-training-and-fine-tuning-LLMs/Module_02_Install_Requirements.ipynb)**: A notebook to install necessary dependencies for this module.
- **[Module_02_LC1_Pre-training_GPT-2_on_Custom_Data_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-02-Pre-training-and-fine-tuning-LLMs/Solutions/Module_02_LC1_Pre-training_GPT-2_on_Custom_Data_Solutions.ipynb)**: Hands-on exercise for pre-training GPT-2 on custom datasets, focusing on how LLMs can start learning patterns on unsupervised datasets but can still be used to perform tasks like Q&A
- **[Module_02_LC2_Full-fine-tuning_BERT_for_Classification_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-02-Pre-training-and-fine-tuning-LLMs/Solutions/Module_02_LC2_Full-fine-tuning_BERT_for_Classification_Solutions.ipynb)**: Covers the process of fully fine-tuning a BERT model for classification tasks, with practical examples.

---

### Module 03: Parameter Efficient Fine-tuning of LLMs

<img align="left" width="150" src="https://i.imgur.com/Nfir5uu.png"/> <br/> In this module, you'll learn about Parameter Efficient Fine-Tuning (PEFT) techniques, which allow you to fine-tune large language models (LLMs) with fewer resources. 
The module includes practical exercises on fine-tuning a BERT model using PEFT methods like QLoRA (Quantized Low-Rank Adaptation) for text classification and named entity recognition. 
You'll also explore advanced methods like merging and switching LoRA adapters on the same model, making your fine-tuning efforts more flexible and efficient.

<br/>

- **[Module_03_Install_Requirements.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-03-Parameter-Efficient-Fine-tuning-LLMs/Module_03_Install_Requirements.ipynb)**: A notebook to set up the environment for parameter efficient fine-tuning.
- **[Module_03_LC1_Parameter-Efficient_fine-tuning_BERT_for_Classification_with_QLoRA_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-03-Parameter-Efficient-Fine-tuning-LLMs/Solutions/Module_03_LC1_Parameter-Efficient_fine-tuning_BERT_for_Classification_with_QLoRA_Solutions.ipynb)**: Exercise on paramter efficient fine-tuning BERT for text classification tasks using QLoRA, allowing for resource-efficient model adaptation.
- **[Module_03_LC2_Parameter-Efficient_fine-tuning_BERT_for_Named_Entity_Recognition_QLoRA_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-03-Parameter-Efficient-Fine-tuning-LLMs/Solutions/Module_03_LC2_Parameter-Efficient_fine-tuning_BERT_for_Named_Entity_Recognition_QLoRA_Solutions.ipynb)**: Focuses on using QLoRA for fine-tuning BERT in named entity recognition (NER) tasks, demonstrating how to apply parameter-efficient methods to complex NLP tasks.
- **[Module_03_LC3_Parameter-Efficient_fine-tuning_Switching_LoRA_Adapters_Walkthrough.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-03-Parameter-Efficient-Fine-tuning-LLMs/Module_03_LC3_Parameter-Efficient_fine-tuning_Switching_LoRA_Adapters_Walkthrough.ipynb)**: Walkthrough of switching LoRA adapters to demonstrate flexible and efficient model fine-tuning across various tasks.

---

### Module 04: Instruction Fine-tune LLMs with Supervised Fine-tuning

<img align="left" width="200" src="https://i.imgur.com/Xr8TdVN.png"/> This module deep dives into instruction-based fine-tuning of LLMs by leveraging Supervised fine-tuning. You'll start by working on a simple, yet challenging task of instruct fine-tuning a TinyLlama 1B SLM for text-to-SQL operations.
Then you will work step-by-step on building a custom fine-tuned RAG pipeline. You will start by leveraging LLMs and prompting to prepare your own labeled datasets for Retrieval-Augmented Generation (RAG). 
Next you will work on fine-tuning your own embedder model using your prepared dataset. Then you will fine-tune a Llama 3 LLM on your labeled dataset for RAG Q&A response generation using instruction tuning and PEFT. 
Finally you will leverage all these components and build your own end-to-end custom fine-tuned RAG pipeline.

<br/>

- **[Module_04_Install_Requirements.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Module_04_Install_Requirements.ipynb)**: Setup notebook for installing dependencies required in this module.
- **[Module_04_LC1_Supervised_Fine_tuning_TinyLlama_1B_for_Text2SQL_Solutions.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Solutions/Module_04_LC1_Supervised_Fine_tuning_TinyLlama_1B_for_Text2SQL_Solutions.ipynb)**: Supervised fine-tuning exercise of TinyLlama 1B for converting SQL table schema and text instruction prompts into SQL queries using instruction-based supervised fine-tuning.
- **[Module_04_LC2_Dataset_Preparation_for_RAG_fine-tuning_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Module_04_LC2_Dataset_Preparation_for_RAG_fine-tuning_Exercise.ipynb)**: A guide to preparing labeled context-question-answer datasets for fine-tuning Retrieval-Augmented Generation (RAG) embedder and instruct models.
- **[Module_04_LC3_Fine-tune_Embedder_Model_for_RAG_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Module_04_LC3_Fine-tune_Embedder_Model_for_RAG_Exercise.ipynb)**: Exercise on fine-tuning an embedder model specifically for RAG tasks, optimizing the model for better retrieval performance.
- **[Module_04_LC4_Supervised_Fine_tuning_Llama_3_LLM_for_RAG_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Module_04_LC4_Supervised_Fine_tuning_Llama_3_LLM_for_RAG_Exercise.ipynb)**: Fine-tuning Llama 3 8B LLM for RAG with instruction-based supervised fine-tuning, enhancing its ability to generate accurate responses based on retrieved information.
- **[Module_04_LC5_Building_Custom_RAG_Systems_with_Fine_tuned_Models_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-04-Instruction-Fine-tuning-LLMs-with-Supervised-Fine-tuning/Module_04_LC5_Building_Custom_RAG_Systems_with_Fine_tuned_Models_Exercise.ipynb)**: Comprehensive exercise on building a custom end-to-end RAG system using the above fine-tuned models.

---

### Module 05: Aligning Fine-tuned LLMs to Human Preferences - RLHF & Beyond

<img align="left" width="200" src="https://i.imgur.com/wDNxEWE.png"/> The final module focuses on aligning fine-tuned LLMs with human preference data using advanced reinforcement learning alignment and reinforcement learning free alignment techniques. You'll explore Reinforcement Learning with Human Feedback (RLHF) and Proximal Policy Optimization (PPO) by aligning LLMs like GPT-2 to generate positive content. The module also covers RL free methods like Direct Preference Optimization (DPO) and Odds-Ratio Preference Optimization (ORPO) to align LLMs like Llama 3 with human preference data, ensuring that your models are helpful, harmless and algined to human preferences.

<br/>

- **[Module_05_Install_Requirements.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-05-Aligning-Fine-tuned-LLMs-with-Human-Preferences-RLHF-PPO-DPO-ORPO/Module_05_Install_Requirements.ipynb)**: Installs the necessary tools and libraries for the exercises in this module.
- **[Module_05_Aligning_GPT2_to_Positive_Content_Generation_with_RLHF_and_PPO_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-05-Aligning-Fine-tuned-LLMs-with-Human-Preferences-RLHF-PPO-DPO-ORPO/Module_05_Aligning_GPT2_to_Positive_Content_Generation_with_RLHF_and_PPO_Exercise.ipynb)**: Exercise on aligning GPT-2 for generating positive content using Reinforcement Learning with Human Feedback (RLHF) and Proximal Policy Optimization (PPO).
- **[Module_05_Aligning_Llama_3_LLM_with_human_preferences_using_DPO_Exercise.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-05-Aligning-Fine-tuned-LLMs-with-Human-Preferences-RLHF-PPO-DPO-ORPO/Module_05_Aligning_Llama_3_LLM_with_human_preferences_using_DPO_Exercise.ipynb)**: Aligns Llama 3 LLM with human preferences using Direct Preference Optimization (DPO).
- **[Module_05_Aligning_Llama_3_LLM_with_human_preferences_using_ORPO_Walkthrough.ipynb](https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024/blob/main/Module-05-Aligning-Fine-tuned-LLMs-with-Human-Preferences-RLHF-PPO-DPO-ORPO/Module_05_Aligning_Llama_3_LLM_with_human_preferences_using_ORPO_Walkthrough.ipynb)**: A detailed walkthrough on using Odds Ratio Preference Optimization (ORPO) for aligning Llama 3 LLM.

---

## Citation

```bibtex
@misc{dipanjan-sarkar-mastering-llms-workshop-2024,
  author = {Dipanjan (DJ) Sarkar},
  title = {Mastering Large Language Models: Training and Fine-Tuning Large Language Models Workshop},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dipanjanS/training-fine-tuning-large-language-models-workshop-dhs2024}}
}
```

## References and Acknowledgements

- [HuggingFace Transformer Documentation](https://huggingface.co/docs/transformers/tasks/token_classification) for giving me datasets and inspiration for the classification and NER tasks
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft/developer_guides/lora) for amazing references to training, merging and switching LoRA Adapters
- [Jay Alammar's amazing blogs](https://jalammar.github.io/illustrated-transformer/) for providing useful info and illustrations to explain these models in a short time especially for a workshop
- [Maxime Labonne's Blog and LLM Course](https://mlabonne.github.io/blog/) for a treasure trove of articles and hands-on examples on supervised fine-tuning and human alignment tuning
- [Sentence Transformers Documentation](https://sbert.net/docs/sentence_transformer/training_overview.html) for really easy to understand examples on how to fine-tune embedder models
- [Raghav Bali and his LLM Workshop Resources](https://github.com/raghavbali/llm_workshop) for giving me a nice starter to build the GPT-2 RLHF tuning notebook which is way better than the default version in HuggingFace docs
- [ArXiV papers](https://arxiv.org/) for easy access to RLHF, PPO, DPO and ORPO papers
- [Analytics Vidhya & GenAI Pinnacle](https://www.analyticsvidhya.com/genaipinnacle) for hosting the conference and some amazing resources I was able to use for covering the concepts around LLMs and fine-tuning
- TBA soon...
