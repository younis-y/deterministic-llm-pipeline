Drop your CV variant files here, named to match the CVVariant enum:

    CV_AI-LLM-Engineering.tex        CV_Quant-Trading.tex             CV_DataScience-Gulf.tex
    CV_EnergySystems-Modelling.tex   CV_Research-DeepLearning.tex     CV_Consulting-Analytics.tex

Filenames are case- and separator-insensitive: `cv_ai_llm_engineering.tex` matches
`CV_AI-LLM-Engineering`. .tex, .md and .txt all work; LaTeX is stripped to plain
text before it reaches the prompt. Run `rolescan cvs` to check they parse.

Without these the scorer still runs, but tailoring advice stays generic because
the model cannot see what each variant actually says.
