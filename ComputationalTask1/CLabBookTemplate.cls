% class for writing an extended essay for the IB using the author-date specification of the Chicago manual of style (16th edition)
\ProvidesClass{CLabBookTemplate.cms}

\LoadClass{article}

% fields that user can set (in addition to title, author, etc)
\makeatletter
	\newcommand{\candidatenum}[1]{\def \@candidatenum {#1}}
	\newcommand{\subtitle}[1]{\def \@subtitle {#1}}
	\newcommand{\subject}[1]{\def \@subject {#1}}
	\newcommand{\wordcount}[1]{\def \@wordcount {#1}}
	\newcommand{\session}[1]{\def \@session {#1}}
	\newcommand{\component}[1]{\def \@component {#1}}
	\candidatenum{}
	\subtitle{}
	\subject{}
	\wordcount{}
	\session{}
	\component{}
\makeatother

% alignment
%\raggedright

% margins
\usepackage[top=1in,bottom=1in,left=1in,right=1in]{geometry}
%symbol list
\RequirePackage{nomencl}
\makenomenclature
\newcommand{\nomunit}[1]{%
\renewcommand{\nomentryend}{\dotfill#1}}

% math
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}

% indentation
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}

% spacing
\usepackage{setspace}
\onehalfspacing %use onehalfspacing in the document

%bibliography
\newcommand{\printmybibliography}{
	\renewcommand{\bibsetup}{\thispagestyle{fancy}} %use the same headers in the bibliography as for the main text	
	\singlespacing %use single spacing in bibliography
	\printbibliography[heading=bibintoc]
	\onehalfspacing	 %change back to onehalfspacing
}

\newcommand{\mytableofcontents}{
	\singlespacing
	\tableofcontents
	\onehalfspacing
}

% figures
\usepackage{graphicx} % For images
\usepackage{float}    % For tables and other floats
\usepackage{subfig}   % For subfigures

% quotations
\usepackage{csquotes}

% links
\usepackage[usenames]{xcolor} % For colors and names
\usepackage[pdftex]{hyperref} % For hyperlinks and indexing the PDF
\hypersetup{ % play with the different link colors here
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=black % set to black to prevent printing blue links
}

% bibliography
%\usepackage[notes,backend=biber]{biblatex-chicago} %for the notes and bibliography style
\usepackage[authordate,backend=biber]{biblatex-chicago} %for the author date style

% page headings and numbering
\makeatletter
	\usepackage{fancyhdr}
	\pagestyle{fancy}
	\renewcommand{\footrulewidth}{\headrulewidth}
%	\renewcommand{\headheight}{15pt}	
	\fancyhead{}
	\fancyfoot{}
	
	\newcommand{\updateheaders}{
		
	%	\ifx \@candidatenum \@empty
	%		\relax
	%	\else
	%		\fancyhead[R]{Candidate \#\@candidatenum}
	%	\fi
		
		\ifx \@session \@empty
			\relax
		\else
			\fancyhead[L]{\@session}
		\fi

		\ifx \@component \@empty
			\relax
		\else
			\fancyhead[C]{\@component}
		\fi
	}

	\fancyfoot[R]{\thepage}
\makeatother

% title
\makeatletter
\def \maketitle {
	\thispagestyle{fancy}
	\fancyhead{}
	\fancyfoot{}
	\fancyfoot[R]{\thepage}		
	\begin{center}
		~\vfill
		\begin{center}
			{\includegraphics[width=80mm]{Figures/leicesteruniversitylogo.png} \\
			\vspace{1.5cm}			
			
			\ifx \@session \@empty
				\relax
			\else
				
				Session: \@session \\
			\fi
			
			\ifx \@subject \@empty
				\relax
			\else
				\@subject \\
			\fi
			
			\ifx \@component \@empty
				\relax
			\else
				\@component \\
			\fi			
		
			\vspace{2cm}			
								
			\Large \@title} \\
			\ifx \@subtitle \@empty
				\relax
			\else
				{\large \@subtitle} \\
			\fi
			
			\vspace{1cm}											
		
			\ifx @author \@empty
				\relax
			\else
				{by \\
				\large \@author} \\
				\ifx \@candidatenum \@empty
					\relax
				\else
					Candidate \#\@candidatenum \\
				\fi
			\fi
			
			\vspace{6cm}						
		
			\ifx \@wordcount \@empty
				\relax
			\else
				Word Count: \@wordcount \\
			\fi
				
			\@date
		\end{center}
		\vfill
		\newpage
		\updateheaders
	\end{center}
}
\makeatother

% abstract
\newenvironment{myAbstract}
{\addcontentsline{toc}{section}{Abstract} % include this in the table of contents
\ \bigskip
\begin{center}
{\bf \Large Abstract}
\medskip
\end{center}}
{\newpage}

% acknowledgment
\newenvironment{myAcknowledgment}
{\addcontentsline{toc}{section}{Acknowledgment} % include this in the table of contents
\ \bigskip
\begin{center}
{\bf Acknowledgment}
\medskip
\end{center}}
{\newpage}