# Expoentes de Lyapunov: quantificando a sensibilidade às condições iniciais

Materiais de apoio para a aula **“Expoentes de Lyapunov: quantificando a sensibilidade às condições iniciais”**, da disciplina **Introdução ao Caos (4300320)**, no Instituto de Física da Universidade de São Paulo (USP).

O repositório reúne o plano de aula, a atividade proposta, os slides, figuras e códigos utilizados para gerar as animações apresentadas durante a aula.

## Organização

```text
.
├── README.md
├── plano_de_aula.pdf
├── atividade_proposta.pdf
├── aula_expoentes_lyapunov.pdf
├── figuras/
│   ├── figura.py
│   └── figura.png
└── videos/
    ├── sensibilidade_lorenz_10_condicoes_iniciais.py
    └── espectro_lyapunov_lorenz.py
```

## Materiais

- **Plano de aula** — `plano_de_aula.pdf`: planejamento, objetivos, desenvolvimento e referências.
- **Atividade proposta** — `atividade_proposta.pdf`: atividade de investigação de ferramentas complementares para caracterização do caos, utilizando o mapa logístico.
- **Slides** — `slides/aula_expoentes_lyapunov.pdf`: apresentação completa da aula.
- **Figuras** — `figuras/`: script Python e imagem utilizados na apresentação.
- **Vídeos e simulações** — `videos/`: códigos para gerar as animações descritas abaixo.

### Vídeos

- `sensibilidade_lorenz_10_condicoes_iniciais.py`: evolução de dez trajetórias com condições iniciais próximas no sistema de Lorenz, ilustrando a sensibilidade às condições iniciais.
- `espectro_lyapunov_lorenz.py`: integração do sistema de Lorenz e das equações variacionais, com ortonormalização periódica para estimar o espectro de Lyapunov.


## Referências

Os livros e artigos abaixo são indicados como referências para o conteúdo da aula. Os arquivos protegidos por direitos autorais não são redistribuídos neste repositório; são disponibilizados links para as páginas dos editores ou para os registros DOI.

### Livros

1. **Strogatz, S. H.** *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering*. 2nd ed. Westview Press, 2015.  
  
2. **Alligood, K. T.; Sauer, T. D.; Yorke, J. A.** *Chaos: An Introduction to Dynamical Systems*. Springer, 1996.  
  
### Artigos

3. **Benettin, G.; Galgani, L.; Giorgilli, A.; Strelcyn, J.-M.** Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems: a method for computing all of them. Part I: Theory. *Meccanica*, 15, 9–20 (1980).  
   [https://doi.org/10.1007/BF02128236](https://doi.org/10.1007/BF02128236).

4. **Benettin, G.; Galgani, L.; Giorgilli, A.; Strelcyn, J.-M.** Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems: a method for computing all of them. Part II: Numerical application. *Meccanica*, 15, 21–30 (1980).  
   [https://doi.org/10.1007/BF02128237](https://doi.org/10.1007/BF02128237).

## Autor
**Bruno R. R. Boaretto**  
---

*Material didático preparado para fins acadêmicos e educacionais.*
