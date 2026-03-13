Hello, 
You are welcome to this little Automatic Differentiation (AD) tool using Julia - Dual Number. This is a homework assignment assigned by Dr. Hayri Sezer of the Department of Mechanical. Dr. Sezer is a faculty member in the Department of Mechanical Engineering at Georgia Southern University, Statesboro Campus, with a research area in Fuel cells, batteries, Fire dynamics, and combustion. You can read more about his work at **https://scholar.google.com/citations?user=eu\_kVXMAAAAJ\&hl=en** 


**While I am still learning. I want to acknowledge that I did not write all the code on my own. I sought support of artificial intelligence and other tools and text. And the code may be updated at some point or another with new things added to it. So you can always get an improved version.** 


DualNumbers.jl is a Julia module that implements forward-mode automatic differentiation (AD) using dual numbers.  

It defines a parametric `Dual` number type, overloads arithmetic and elementary mathematical functions, and provides a simple interface for computing first and higher-order derivatives of scalar functions.



The module is designed as a lightweight educational and research tool for understanding the mathematical foundations of automatic differentiation and for prototyping numerical algorithms requiring derivative computation.



---



\# Table of Contents



\- \[Overview](#overview)

\- \[Mathematical Background](#mathematical-background)

\- \[Features](#features)

\- \[Installation](#installation)

\- \[Usage](#usage)

\- \[Core Type](#core-type)

\- \[Supported Operations](#supported-operations)

\- \[Elementary Functions](#elementary-functions)

\- \[Derivative Computation](#derivative-computation)

\- \[Examples](#examples)

\- \[Project Structure](#project-structure)

\- \[Testing](#testing)

\- \[Limitations](#limitations)

\- \[Future Improvements](#future-improvements)

\- \[License](#license)

\- \[Author] (#author)



---



\# Overview



Automatic differentiation computes derivatives exactly to machine precision by applying the chain rule at the level of elementary operations.



Unlike:



\- Finite differences – approximate derivatives and suffer from truncation errors

\- Symbolic differentiation – may produce large symbolic expressions



automatic differentiation propagates derivatives during evaluation of numerical code.



`DualNumbers.jl` implements this approach using dual numbers, allowing derivatives to be computed automatically when functions are evaluated.



---



\# Mathematical Background



A dual number has the form



\\\[

x + x'\\varepsilon

\\]



where



\- \\(x\\) = function value

\- \\(x'\\) = derivative

\- \\(\\varepsilon^2 = 0\\)



When a function is evaluated at a dual number:



\\\[

f(x + \\varepsilon) = f(x) + f'(x)\\varepsilon

\\]



Thus the derivative appears automatically in the coefficient of \\(\\varepsilon\\).



Example:



\\\[

f(x) = x^2

\\]



Evaluate at



\\\[

x + \\varepsilon

\\]



\\\[

(x+\\varepsilon)^2 = x^2 + 2x\\varepsilon

\\]



Therefore:



\- function value = \\(x^2\\)

\- derivative = \\(2x\\)



This is the basis of \*\*forward-mode automatic differentiation\*\*.



---



\# Features



\- Parametric \*\*Dual number type\*\*

\- Subtype of Julia's `Number`

\- Automatic type promotion

\- Mixed scalar–dual arithmetic

\- Operator overloading for



&nbsp; - `+`

&nbsp; - `-`

&nbsp; - `\*`

&nbsp; - `/`

&nbsp; - `^`



\- Supported elementary functions



&nbsp; - `sin`

&nbsp; - `cos`

&nbsp; - `tan`

&nbsp; - `exp`

&nbsp; - `sqrt`

&nbsp; - `log`

&nbsp; - `asin`

&nbsp; - `acos`



\- Hyperbolic functions



&nbsp; - `sinh`

&nbsp; - `cosh`

&nbsp; - `tanh`



\- Comparison operators



&nbsp; - `==`

&nbsp; - `<`

&nbsp; - `<=`



\- Higher-order derivatives via repeated differentiation



---



\# Installation



**## Option 1 — Local Module** 

Save the file as:



**DualNumbers.jl**



Then load it:



```julia

include("DualNumbers.jl")

using .DualNumbers

## **Option 2 — As a Julia Package**



Recommended directory structure:



DualNumbers/

│

├── Project.toml

├── README.md

├── src

│   └── DualNumbers.jl

└── test

&nbsp;   └── runtests.jl



Load the package:



</> julia
using DualNumbers





**Usage**



Define a function and compute derivatives.



using DualNumbers



f(x) = x^3 + 2x + 1



dual\_diff\_n(f, 2.0, 1)



Output:



14

Core Type



The module defines the parametric type:



struct Dual{T<:Number} <: Number

&nbsp;   val::T

&nbsp;   der::T

end

Fields

Field	Meaning

val	function value

der	derivative



Example:



x = Dual(3.0,1.0)

represents

3+ε





**Supported Operations**



The module overloads arithmetic operations.



**Addition**

(a+bε)+(c+dε)=(a+c)+(b+d)ε



**Subtraction**

(a+bε)−(c+dε)=(a−c)+(b−d)ε



**Multiplication**

(a+bε)(c+dε)=ac+(ad+bc)ε



**Division**

&nbsp;	​



**Power Rules**



Dual base:



Scalar base:



**Elementary Functions**



The module overloads several functions.



Function	Derivative

sin(x)		cos(x)

cos(x)		-sin(x)

tan(x)		sec²(x)

exp(x)		exp(x)

sqrt(x)		1/(2√x)

log(x)		1/x



Inverse functions:



Function	Derivative

asin(x)		1/√(1-x²)

acos(x)		-1/√(1-x²)



Hyperbolic functions:



Function	Derivative

sinh(x)	cosh(x)

cosh(x)	sinh(x)

tanh(x)	1 − tanh²(x)

Derivative Computation

First Derivative



Internal helper:

</> julia

dual\_diff\_1(f,x)



Computes

f'(x)



by evaluating



f(Dual(x,1))

Higher-Order Derivatives



Main user function:



dual\_diff\_n(f,x,n)



Computes the n-th derivative



dual\_diff\_n(f,x,0) → f(x)

dual\_diff\_n(f,x,1) → f'(x)

dual\_diff\_n(f,x,2) → f''(x)



Example:



f(x) = x^4



dual\_diff\_n(f,3.0,1)

dual\_diff\_n(f,3.0,2)



Results:



108

108



because

f'(x)=4x^3

f''(x)=12x^2





Examples

Polynomial Differentiation

f(x) = x^3 + 2x^2 + x



dual\_diff\_n(f,2.0,1)



Result



21

Trigonometric Function

f(x) = sin(x)



dual\_diff\_n(f,pi/4,1)



Expected



cos(pi/4)

Composite Function

f(x) = exp(sin(x) + x^2)



dual\_diff\_n(f,1.0,1)



The chain rule is applied automatically.



Direct Dual Evaluation

x = Dual(2.0,1.0)



y = x^3 + sin(x)



println(y)



Output format:



Dual(value, derivative)

Project Structure



Recommended structure for GitHub repository.



DualNumbers.jl

│

├── README.md

├── Project.toml

│

├── src

│   └── DualNumbers.jl

│

└── test

&nbsp;   └── runtests.jl

Testing



Example test file.



test/runtests.jl



using Test

using DualNumbers



@test dual\_diff\_n(x->x^2,3.0,1) ≈ 6

@test dual\_diff\_n(x->x^3,2.0,1) ≈ 12

@test dual\_diff\_n(x->sin(x),0.0,1) ≈ 1

@test dual\_diff\_n(x->exp(x),0.0,1) ≈ 1



Run tests with



Pkg.test()

Limitations



**Current limitations include:**



Only scalar functions supported



Limited elementary function coverage



Higher-order derivatives implemented via repeated differentiation



No Jacobian or gradient utilities



No vector automatic differentiation







**Future Improvements**



Possible extensions:



nested dual numbers for efficient higher-order derivatives



vector-valued differentiation



Jacobian and Hessian computation



additional functions (atan, abs, etc.)



GPU compatibility



performance optimization



documentation generation using Documenter.jl



License



MIT License





Author

Ayobami Samuel Akinfenwa

PhD Researcher — Mechanical Engineering

Georgia Southern University



Research Area:



Human-Robot Interaction



Biomedical Signal Processing



Multimodal AI Systems



Scientific Computing











































































