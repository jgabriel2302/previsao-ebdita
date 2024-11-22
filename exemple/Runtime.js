class FinanceData {
    periodo = "";
    ano = 0;
    trimestre = 0;
    receitaLiquida = 0;
    cpv = 0;
    lucroBruto = 0;
    despesasOperacionais = 0;
    ebitda = 0;
    margemBruta = 0; // porcentagem
    taxaCambio = 0;

    constructor(object) {
        if (typeof object === "object" && object !== null) {
            if (object.Periodo && typeof object.Periodo === "string")
                this.periodo = object.Periodo;

            if (object.Ano && typeof object.Ano === "number")
                this.ano = object.Ano;

            if (object.Trimestre && typeof object.Trimestre === "number")
                this.trimestre = Math.max(Math.min(4, object.Trimestre), 1);

            if (
                object["Receita de líquida"] &&
                typeof object["Receita de líquida"] === "number"
            )
                this.receitaLiquida = object["Receita de líquida"];

            if (object.CPV && typeof object.CPV === "number")
                this.cpv = object.CPV;

            if (
                object["Lucro bruto"] &&
                typeof object["Lucro bruto"] === "number"
            )
                this.lucroBruto = object["Lucro bruto"];

            if (
                object["Despesas operacionais + Depre + Amort"] &&
                typeof object["Despesas operacionais + Depre + Amort"] ===
                "number"
            )
                this.despesasOperacionais =
                    object["Despesas operacionais + Depre + Amort"];

            if (object.EBITDA && typeof object.EBITDA === "number")
                this.ebitda = object.EBITDA;

            if (
                object["Margem Bruta"] &&
                typeof object["Margem Bruta"] === "number"
            )
                this.margemBruta = Math.max(
                    Math.min(1, object["Margem Bruta"]),
                    0
                );

            if (
                object["Taxa de Câmbio"] &&
                typeof object["Taxa de Câmbio"] === "number"
            )
                this.taxaCambio = object["Taxa de Câmbio"];
        }
    }
}

function jsonToTable(json, headersKeys, rowsKeys) {
    let table = document.createElement("table");
    let thead = document.createElement("thead");
    let tbody = document.createElement("tbody");

    let trHeader = document.createElement("tr");
    thead.append(trHeader);

    let th0 = document.createElement("th");
    th0.innerHTML = "#";
    trHeader.append(th0);

    for (const header of headersKeys) {
        let th = document.createElement("th");
        th.innerHTML = header;
        trHeader.append(th);
    }
    for (const row of rowsKeys) {
        let tr = document.createElement("tr");
        let td0 = document.createElement("td");
        td0.innerHTML = row;
        tr.append(td0);
        for (const header of headersKeys) {
            let td = document.createElement("td");
            td.innerHTML = Intl.NumberFormat("pt-BR", {
                style: "decimal",
            }).format(json[header][row]);
            tr.append(td);
        }
        tbody.append(tr);
    }
    table.append(thead, tbody);
    document.querySelector("#result").append(table);
}

(function () {
    document.querySelector('#status').innerHTML += `<div class='system'><b>${new Date().toTimeString().substring(0, 8)}</b> Obtendo dados de <i>dadosPetro.json</i></div>`
    fetch("./dadosPetro.json")
        .then((response) => response.json())
        .then((dados) => {
            document.querySelector('#status').innerHTML += `<div class='system'><b>${new Date().toTimeString().substring(0, 8)}</b> Formatando dados</div>`
            const trimestres = {};
            const KPIs = {};
            for (const dado of dados.data) {
                if (!trimestres[dado.Periodo])
                    trimestres[dado.Periodo] = {
                        Periodo: dado.Periodo,
                        Ano: Number("20" + String(dado.Periodo).substring(2)),
                        Trimestre: Number(String(dado.Periodo).substring(0, 1)),
                    };
                if (!trimestres[dado.Periodo][dado.KPI])
                    trimestres[dado.Periodo][dado.KPI] = dado.Valor;

                if (!KPIs[dado.KPI]) KPIs[dado.KPI] = true;
            }

            document.querySelector('#status').innerHTML += `<div class='system'><b>${new Date().toTimeString().substring(0, 8)}</b> Renderizando dados em formado de tabela</div>`
            jsonToTable(
                trimestres,
                Object.values(trimestres)
                    .sort((a, b) =>
                        a.Ano !== b.Ano ? a.Ano - b.Ano : a.Trimestre - b.Trimestre
                    )
                    .map((x) => x.Periodo),
                Object.keys(KPIs)
            );

            document.querySelector('#status').innerHTML += `<div class='system'><b>${new Date().toTimeString().substring(0, 8)}</b> Organizando dados</div>`
            const historico = Object.values(trimestres).map(
                (dado) => new FinanceData(dado)
            );

            document.querySelector('#status').innerHTML += `<div class='system'><b>${new Date().toTimeString().substring(0, 8)}</b> Dados prontos para treino.</div>`

            const network = new NeuralNetwork([6, 8, 1]);
            const attention = new Attention(6); // Mecanismo de atenção para 5 variáveis de entrada

            let trainingCount = 0;

            const DIRECT_NNA = true;

            const train = (keepLearing = false, minMAPE = 0.03, lastMAPE = 1) => {

                const data = NeuralNetwork.applyDropout(historico, 0.4);
                // Passo 1: Normalização dos Dados
                if (trainingCount === 0) document.querySelector('#status').innerHTML += `<div class='train'><b>${new Date().toTimeString().substring(0, 8)}</b> Treino: Normalização dos Dados (usando 40% dos dados histórico)</div>`
                //Entradas
                const receitaLiquida = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.receitaLiquida)
                );
                const cpv = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.cpv)
                );
                const despesasOperacionais = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.despesasOperacionais)
                );
                const taxaCambio = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.taxaCambio)
                );
                const margemBruta = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.margemBruta)
                );
                const trimestre = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.trimestre)
                );
                //Saídas
                const ebitdas = NeuralNetwork.normalizeFeature(
                    data.map((d) => d.ebitda)
                );

                // Passo 2: Construção do Conjunto de Entradas e Saídas
                if (trainingCount === 0) document.querySelector('#status').innerHTML += `<div class='train'><b>${new Date().toTimeString().substring(0, 8)}</b> Treino: Construção do Conjunto de Entradas e Saídas</div>`
                const rawInputs = receitaLiquida.map((_, i) => [
                    receitaLiquida[i],
                    cpv[i],
                    despesasOperacionais[i],
                    taxaCambio[i],
                    margemBruta[i],
                    trimestre[i]
                ]);
                const outputs = ebitdas;

                const attentionResult = DIRECT_NNA ? [] : attention.computeAttention(rawInputs);
                const inputs = DIRECT_NNA ? rawInputs: attentionResult.contextVectors;

                // Passo 4: Treinamento
                if (trainingCount === 0) document.querySelector('#status').innerHTML += `<div class='train'><b>${new Date().toTimeString().substring(0, 8)}</b> Treino: Treinando rede com Entradas e Saídas</div>`
                const learningRate = lastMAPE / 100;
                NeuralNetwork.train(network, inputs, outputs, learningRate);

                const check = () => {
                    // Passo 5: Avaliação
                    if (trainingCount === 0) document.querySelector('#status').innerHTML += `<div class='train'><b>${new Date().toTimeString().substring(0, 8)}</b> Treino: Avaliando treinamento com base nas previsões</div>`
                    const predictions = inputs.map((input) =>
                        Number(NeuralNetwork.feedForward(input, network))
                    );
                    if (trainingCount === 0) document.querySelector('#status').innerHTML += `<div class='train'><b>${new Date().toTimeString().substring(0, 8)}</b> Executando avaliação com MAPE</div>`
                    // Passo 6: Avaliação com MAPE
                    const mape = NeuralNetwork.meanAbsolutePercentageError(
                        outputs,
                        predictions
                    );

                    trainingCount++;
                    if (mape > minMAPE && keepLearing) {
                        const mensagem = `MAPE ${Intl.NumberFormat("pt-BR", {style: "decimal"}).format(mape * 100)}% não é satisfatório. Treinando novamente. (Treino nº ${trainingCount})`;
                        
                        if (trainingCount === 1) document.querySelector('#status').innerHTML += `<div class='train' id="fixed-train"><b>${new Date().toTimeString().substring(0, 8)}</b> ${mensagem}</div>`
                        else document.querySelector('#status #fixed-train').innerHTML = `<b>${new Date().toTimeString().substring(0, 8)}</b> ${mensagem}`;
                        
                        setTimeout(()=>{
                           train(keepLearing, minMAPE, mape);
                        }, 10);
                    } else {
                        document.querySelector('#status #fixed-train').classList.add('mape');
                        document.querySelector('#status #fixed-train').classList.remove('train');
                        document.querySelector('#status #fixed-train').innerHTML = `<b>${new Date().toTimeString().substring(0, 8)}</b> Treino Finalizado. Modelo treinado até a média Absoluta percentual de erro ser menor que ${Intl.NumberFormat("pt-BR", {style: "decimal"}).format(minMAPE * 100)}% (MAPE: ${Intl.NumberFormat("pt-BR", {
                            style: "decimal",
                        }).format(mape * 100)}% em ${trainingCount} treinos)`
                        
                        document.querySelector('#predict').removeAttribute('disabled');

                        const previsao =  NeuralNetwork.feedForward(inputs[0], network);
                        const real = data.map((d) => d.ebitda);
                        console.log({
                            previsao,
                            real: outputs[0],
                            ebitdasPrevisao: NeuralNetwork.unnormalizeFeature(real, previsao[0]),
                            ebitdasReal: NeuralNetwork.unnormalizeFeature(real, outputs[0])
                        })
                    }
                };
                check();
            };

            train(true);
        })
        .catch((err) => {
            console.error(err);
        });
})();