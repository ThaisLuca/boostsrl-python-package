/* Copyright (c) 2017-2018 StARLinG Lab
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 *
 *	A parser for the modes used in boostsrl-python-package, using Antlr4
 *
 *	Author: Alexander L. Hayes (@batflyer)
 *	Email: alexander@batflyer.net
 *
 *	I found this guide to be extremely helpful:
 *	https://tomassetti.me/antlr-mega-tutorial/
 */

grammar Modes;

prog:	(expr NEWLINE)* ;
expr:	expr ('*'|'/')
expr
    |	expr ('+'|'-')
expr
    |   INT
    |   '(' expr ')'
    ;
NEWLINE : [\r\n]+ ;
INT     : [0-9]+ ;