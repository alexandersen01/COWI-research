#  3 møte, 06.02.2025 INF219
**Tilstede:** Thone, Hanna, Håvard, Jakob, Adrian, Hermann, Kyrre, Sverre, Dag

**Neste møte:** Starten av uke 7 , neste møte med COWI 6 mars 14:00 - 16:00.


### Hva er OPPGAVEN / MÅL:
* Plassere flest mulige sirkler på et rom/areal uten overlapp. (sirkler = lys).


### Hva vi har gjort på møtet:
* Jakob forklarte hva han har gjort, og hvorfor han har gjort som han har gjort. 
    * Han bruker brute force, men han går ikke gjennom alle mulige kombinasjoner. Altså fjerner de løsningene som ikke er gode.
    * Han har brukt modulen pulp.  

## Nye krav til neste møte:
* Lage sirkler som er **gradient**, slik at de viser hvor mye de lyser og hvor, og hvor de stopper å lyse.
    * Det finnes en **looks-modell** denne kan vi finne på Google. en arbeidsplass skal ha *500 looks (et rom har krav på **gjevnhet**. Altså ikke bekmørt ett sted og dritlyst ett sted.)* på en arbeidsplate, en koridor skal ha *200 looks*. 
* Vi skal ha **færrest** mulige lamper.
* *Vi kan eventuelt* lage en konkurent til hva Jakob har laget til nå. 

* Vi kan ha to grid, ett grid til lys, og ett grid til møbler, for å se hvor vi skal plassere lys. 
    * Vi kan gi score på hvor lysene blir plassert iforhold til rommet og møblene.
    * **Problemer med møbler:** idag blir det vurdert av et *menneske*, ved data så skilles ikke et bord og en stol. eks: i cowi sitt datasett vil en *plante* bli behandlet som et møbel. Altså ting blir møbel eller ikke møbel. 
    * **Herman** vil at vi skal kunne sette steder manuelt der vi vil at det ikke skal være lys, og hvor vi vil at det skal være lys. 
* Finne ut hvor fort lønsingen vår går til de forsjellige rommene i datasettet.
* *Dialux*, kan vi bruke for å se lysfordeling. 

#### Forutsetninger/problembeskrivelse vi har satt med COWI
Altså hvordan COWI skal klare å **Ranke** løsningene våres:
* Plassere sirkler innenfor ett rom, med best mulig dekning, men uten for mange.
* Kostnad for lysene:
    * Må ha en straff for dårlig mønster, og hvis det er bra så er det *WOHOOOO!*, dette kan vi teste ut for å se hva som er bra eller ikke. 
* Minimum dekningskrav er ikke så farlig, men skal være dekket så og så mye
* Definere *gradienten/lys intensiteten til lyset*.
    * Glomox lyskurve, søk disse opp.






### Spørsmål og svar til COWI
* Trenger en mer direkte definisjon av målet
    - Kan sirklene overlappe, eller ikke? Eller både og.
    - Definere regler, eks:
        - Minst 90% covarage
        - Mest mulig coverage, eller færrest mulig lamper
        * **Kyrre** vil ha færrest mulig sirkler/lamper, men som gir mest mulig lys
        * **Kyrre** vil at vi skal vurdere vær pixel i rommet til å vurdere om vi har *nok* lys i rommet, altså ha en **gradient** sirkel. :D
        
* Har de konkrete regler for hvordan man presenterer andres matriale?
    - De lurer på om det er kjekt for dem å satse på pulp/open sorce? (Kyrre sa det var "dumt spørsmål")

* Hvordan kan vi bygge videre på MVP?

* Hvilke komponenter skal prioriteres, hvordan burde vi skalere videre?
* Hva er reglene for de ulike komponenter
    - Hva er forholdet til de ulike komponentene
    - Spesifikasjon på hvor store sirklene skal være?

### Spørsmål og tanker fra COWI og oss
* Forholde oss til dekningsareal, prosent?
* Påstår de at vi kan få til 100% coverage uten så mange sirkler som kommer opp når Jakob sin løsning viser 100% coverage. 

* De vil at vi skal få lagd noe som kan lære fra løsningene vi har allerede laget. 
* Dele opp gruppen i flere små grupper, som lager forskjellige mulige angrepsvinkler/løsninger.

* Kyrre syntes det hadde vært gøy hvis vi kan "rate" de ulike metodene/algoritmene til nestegang?

* **Kyrre bryr seg om lysene:** dekningsareal per enhet, dette øker prisen. Dette kan være to caser. 

* Kyrre vil at vi skal gjøre denne veldig god, før vi fortsetter med nye komponenter.
    * Hvor fort får vi gode resulteter, og hvor *billig* kan det bli?
    * Lys har gjerne alltid en symetrisk løsning/**mønster**, så lenge det ser vakkert ut.
        * Kan vi introdusere en straff for å bryte dette mønstret?
        * Hvordan kan vi kvantifisere kvaliteten på mønsteret?
        * **VI** skal komme på forslag på hvordan vi kan få lysene til å ha en *symetrisk/vakker* løsning


    * **Kyrre** skal gi oss mer info på rommene, og hva som er inni dem?

* **Kyrre** syntes det er for tidlig til å fokusere på nye komponenter. 

* **Dag** er usikker på om Jakob sin modell er det beste, derfor vil han at vi skal løse det fra scratch på flere mulige måter.

#### Neste krav etter vi er ferdig med løsningen til lys, altså dette skal vi ikke begynne med før om lenge!!!!:
* Ha lys og ventelasjon som neste krav, de skal ikke treffe hverandre. Skal fortsatt ha samme mål om å dekke rommet. 
* Lys fungerer annerledes med tanke på farge på veggene/rommet, dette kan vi ta med senere som et krav.


### Mål til neste møte:
* **ALLE:** Sette seg inn i koden til Jakob sin MVP.  
* **ALLE:** Sette oss inn i Lisens/regler om bruk av andres matriale. (MIT licens?)

* **ALLE:** Gå inn på **SLACK**
* **KYRRE** vil at vi tracker oppgavene i *miro*
    * Vi bruker KANBAN
    * **KYRRE** vil ha nettsiden til Jakob i *miro*

* **Thone** Definer alt vi har blit enige om i *miro*






