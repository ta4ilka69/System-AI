# Лабораторная работа №1

## Предметная область: инструменты и крафт в игре Minecraft

[Реализация на Prolog](./db.pl)

Необходимо было реализовать факты с 1 и 2 аргументами:

    block - добываемый блок (факт с 1 аргументом)
    item - предмет, который можно получить из блока (факт с 1 аргументом)
    tool - инструмент для добычи блока (факт с 1 аргументом)
    crafting - рецепт крафта или добычи (факт с 2 аргументами)
    endurance - прочность блока (факт с 2 аргументами)
    damage - урон инструмента (факт с 2 аргументами)
    mining - минимальный необходимый урон инструмента для добычи блока (факт с 2 аргументами)

Кроме того, требовалось реализовать правила на основе фактов:
    
    - Что можно скрафтить, имея текущие ресурсы?
    - Можно ли добыть блок инструментом?
    - сколько времени нужно на добычу блока?
    - Какие блоки можно добыть инструментом?
    - Какими инструментами можно добыть блок?

## Примеры использования

Какие блоки существуют в игре?

```prolog
3 ?- block(Block).
Block = wood ;
Block = stone ;
Block = iron_ore ;
Block = diamond ;
Block = gold_ore ;
Block = obsidian ;
Block = ancient_blocks.
```

Сколько нужно времени, чтобы добыть блок obsidian, имея кирку из алмазов?

```prolog
3 ?- time_to_mine(obsidian, diamond_kirk, Time).
Time = 5.
```

Какие блоки можно добыть железной киркой?

```prolog
4 ?- blocks_mineable_with(iron_kirk,Blocks).
Blocks = [wood, stone, iron_ore, diamond, gold_ore].
```

Каким инструментом можно добыть блок gold_ore?

```prolog
tools_to_mine(gold_ore, Tools).
Tools = [iron_kirk, diamond_kirk, gold_kirk, netherite_kirk].
```

Что можно создать, имея блок дерева, камня и железной руды?

```prolog
6 ?- can_craft(Tools, [wood,stone,iron_ore]). 
Tools = wooden_kirk ;
Tools = rock_kirk ;
Tools = iron_kirk ;
Tools = iron_ingot ;
Tools = wood_boards ;
Tools = stick ;
```

Что можно создать, добывая блоки с помощью кирки из железа за время от 2 до 5 включительно?

```prolog
7 ?- blocks_mineable_with(iron_kirk, Blocks),findall(Block, (member(Block, Blocks), time_to_mine(Block, iron_kirk, Hits), Hits >= 2, Hits =< 5), Result).
Blocks = [wood, stone, iron_ore, diamond, gold_ore],
Result = [iron_ore, diamond, gold_ore].
```

Какие блоки имеют прочность меньше 20 или больше 40?

```prolog
11 ?- block(Block), endurance(Block, Endurance), (Endurance > 40 ; Endurance < 20).
Block = wood,
Endurance = 10 ;
Block = diamond,
Endurance = 50 ;
Block = obsidian,
Endurance = 60 ;
Block = ancient_blocks,
Endurance = 60.
```


