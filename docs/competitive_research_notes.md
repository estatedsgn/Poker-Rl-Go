# Competitive NLHE notes (what strong systems do)

Ниже собрал практичные ориентиры из публичных материалов/статей, чтобы мы шли в сторону соревновательной силы.

## 1) Что показывают сильные работы

1. **DeepStack (2017)**
   - Идея: continual re-solving в подиграх + value network как аппроксимация leaf values.
   - Практический вывод для нас: на поздних улицах нужно качественное значение состояния (value head / equity features), а не только policy head.

2. **Libratus (2017)**
   - Идея: blueprint стратегия + nested subgame solving + дообучение против эксплойтов.
   - Практический вывод: нужна смесь оффлайн pretraining + онлайн/итеративный self-play с обновлением пула оппонентов.

3. **Pluribus (2019)**
   - Идея: multi-player NLHE, self-play, поиск в ограниченном пространстве действий.
   - Практический вывод: action abstraction и дисциплина evaluation против пула оппонентов критичны, особенно для 6-max и больше.

4. **NFSP / Deep CFR / ReBeL**
   - Идея: приближение к равновесным стратегиям через self-play и regret/value decomposition.
   - Практический вывод: PPO baseline нормален для старта, но для топ-уровня стоит рассмотреть CFR-подобные/гибридные методы.

## 2) Практические гайды/репозитории, на которые обычно ориентируются

- **OpenSpiel (DeepMind):** алгоритмы CFR/NFSP/Policy Gradient и примеры покера.
- **RLCard:** быстрый baseline для карточных игр и простого RL-пайплайна.
- **PokerRL:** исследовательский фреймворк для self-play в покере (полезен как reference архитектуры).

## 3) Совпадаем ли мы с best practices?

Короткий ответ: **частично да, но до “соревновательного” уровня еще шаги нужны**.

Что уже в репозитории в правильном направлении:
- action masking + legal action mapping;
- расширенные pot-based raises и all-in;
- ограничение до 8 игроков;
- equity estimation;
- actor-critic loss + backprop utilities.

Что надо добавить следующим этапом:
1. Полноценный игровой движок/среда с корректным betting state machine.
2. Multi-agent self-play league (snapshot pool).
3. Evaluation harness: BB/100 на 100k+ раздач против фиксированного пула.
4. Опционально: переход на CFR-like гибрид или ReBeL-style value/re-solving для поздних улиц.

## 4) Рекомендуемый ближайший план (реально выполнимый)

1. Подключить `CompetitiveEnvConfig` к реальному env backend (RLCard/OpenSpiel/pokerkit).
2. Реализовать `train.py` с PPO + action mask + логирование в TensorBoard/W&B.
3. Сделать `evaluate.py` с фиксированными seed и opponent-pool.
4. Запустить серию экспериментов: 6-max, 100bb, 3–5 разных seed.


## 5) Что добавлено в код для сильной игры уже сейчас

- Реализован `src/self_play_league.py`:
  - хранение snapshot-политик,
  - Elo рейтинг для пула агентов,
  - PFSP-подобный сэмплинг оппонентов (ближайшие по силе + exploration),
  - промоут нового checkpoint в active policy при достаточном отрыве по Elo.

Это соответствует практике population-based self-play и league training из сильных покерных систем.
