import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Stock Tracker Pro - yfinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-price {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .stock-change-positive {
        color: #00cc96;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stock-change-negative {
        color: #ef553b;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .portfolio-table {
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

if 'notifications' not in st.session_state:
    st.session_state.notifications = []

if 'email_config' not in st.session_state:
    st.session_state.email_config = {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email': '',
        'password': ''
    }

# Titre principal
st.markdown("<h1 class='main-header'>📊 Stock Tracker Pro - Analyse en Temps Réel</h1>", unsafe_allow_html=True)

# Sidebar pour la navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
    st.title("Navigation")
    
    menu = st.radio(
        "Choisir une section",
        ["📈 Tableau de bord", 
         "💰 Portefeuille virtuel", 
         "🔔 Alertes de prix",
         "📧 Notifications email",
         "📤 Export des données",
         "🤖 Prédictions ML"]
    )
    
    st.markdown("---")
    
    # Configuration commune
    st.subheader("⚙️ Configuration")
    
    # Liste des symboles
    default_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
    
    # Sélection du symbole principal
    symbol = st.selectbox(
        "Symbole principal",
        options=st.session_state.watchlist + ["Autre..."],
        index=0
    )
    
    if symbol == "Autre...":
        symbol = st.text_input("Entrer un symbole", value="AAPL").upper()
        if symbol and symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
    
    # Période et intervalle
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox(
            "Période",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
    
    with col2:
        interval_map = {
            "1m": "1 minute", "2m": "2 minutes", "5m": "5 minutes",
            "15m": "15 minutes", "30m": "30 minutes", "1h": "1 heure",
            "1d": "1 jour", "1wk": "1 semaine", "1mo": "1 mois"
        }
        interval = st.selectbox(
            "Intervalle",
            options=list(interval_map.keys()),
            format_func=lambda x: interval_map[x],
            index=4 if period == "1d" else 6
        )
    
    # Auto-refresh
    auto_refresh = st.checkbox("Actualisation automatique", value=False)
    if auto_refresh:
        refresh_rate = st.slider(
            "Fréquence (secondes)",
            min_value=5,
            max_value=60,
            value=30,
            step=5
        )

# Fonctions utilitaires
@st.cache_data(ttl=300)
def load_stock_data(symbol, period, interval):
    """Charge les données boursières"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        info = ticker.info
        return hist, info
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None, None

def send_email_alert(subject, body, to_email):
    """Envoie une notification par email"""
    if not st.session_state.email_config['enabled']:
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = st.session_state.email_config['email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(
            st.session_state.email_config['smtp_server'], 
            st.session_state.email_config['smtp_port']
        )
        server.starttls()
        server.login(
            st.session_state.email_config['email'],
            st.session_state.email_config['password']
        )
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Erreur d'envoi: {e}")
        return False

def check_price_alerts(current_price, symbol):
    """Vérifie les alertes de prix"""
    triggered = []
    for alert in st.session_state.price_alerts:
        if alert['symbol'] == symbol:
            if alert['condition'] == 'above' and current_price >= alert['price']:
                triggered.append(alert)
            elif alert['condition'] == 'below' and current_price <= alert['price']:
                triggered.append(alert)
    
    return triggered

# Chargement des données
hist, info = load_stock_data(symbol, period, interval)

if hist is not None and not hist.empty:
    current_price = hist['Close'].iloc[-1]
    
    # Vérification des alertes
    triggered_alerts = check_price_alerts(current_price, symbol)
    for alert in triggered_alerts:
        st.balloons()
        st.success(f"🎯 Alerte déclenchée pour {symbol} à ${current_price:.2f}")
        
        # Notification email
        if st.session_state.email_config['enabled']:
            subject = f"🚨 Alerte prix - {symbol}"
            body = f"""
            <h2>Alerte de prix déclenchée</h2>
            <p><b>Symbole:</b> {symbol}</p>
            <p><b>Prix actuel:</b> ${current_price:.2f}</p>
            <p><b>Condition:</b> {alert['condition']} ${alert['price']:.2f}</p>
            <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            send_email_alert(subject, body, st.session_state.email_config['email'])
        
        # Retirer l'alerte si elle est à usage unique
        if alert.get('one_time', False):
            st.session_state.price_alerts.remove(alert)

# ============================================================================
# SECTION 1: TABLEAU DE BORD
# ============================================================================
if menu == "📈 Tableau de bord":
    # Métriques principales
    st.subheader(f"📊 Aperçu en temps réel - {symbol}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
    change = current_price - previous_close
    change_pct = (change / previous_close) * 100
    
    with col1:
        st.metric(
            label="Prix actuel",
            value=f"${current_price:.2f}",
            delta=f"{change:.2f} ({change_pct:.2f}%)"
        )
    
    with col2:
        day_high = hist['High'].iloc[-1]
        st.metric("Plus haut", f"${day_high:.2f}")
    
    with col3:
        day_low = hist['Low'].iloc[-1]
        st.metric("Plus bas", f"${day_low:.2f}")
    
    with col4:
        volume = hist['Volume'].iloc[-1]
        volume_formatted = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
        st.metric("Volume", volume_formatted)
    
    # Graphique principal
    st.subheader("📉 Évolution du prix")
    
    fig = go.Figure()
    
    # Chandeliers ou ligne selon l'intervalle
    if interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Prix',
            increasing_line_color='#00cc96',
            decreasing_line_color='#ef553b'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Prix',
            line=dict(color='#1f77b4', width=2)
        ))
    
    # Ajouter les moyennes mobiles
    ma_20 = hist['Close'].rolling(window=20).mean()
    ma_50 = hist['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=ma_20,
        mode='lines',
        name='MA 20',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=ma_50,
        mode='lines',
        name='MA 50',
        line=dict(color='purple', width=1, dash='dash')
    ))
    
    # Volume
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color='lightgray', opacity=0.3)
    ))
    
    fig.update_layout(
        title=f"{symbol} - {period}",
        yaxis_title="Prix ($)",
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis_title="Date",
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informations sur l'entreprise
    with st.expander("ℹ️ Informations sur l'entreprise"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Nom :** {info.get('longName', 'N/A')}")
            st.write(f"**Secteur :** {info.get('sector', 'N/A')}")
            st.write(f"**Industrie :** {info.get('industry', 'N/A')}")
            st.write(f"**Site web :** {info.get('website', 'N/A')}")
        
        with col2:
            st.write(f"**Capitalisation :** ${info.get('marketCap', 0):,.0f}")
            st.write(f"**P/E :** {info.get('trailingPE', 'N/A')}")
            st.write(f"**Dividende :** {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "**Dividende :** N/A")
            st.write(f"**Beta :** {info.get('beta', 'N/A')}")

# ============================================================================
# SECTION 2: PORTEFEUILLE VIRTUEL
# ============================================================================
elif menu == "💰 Portefeuille virtuel":
    st.subheader("💰 Gestion de portefeuille virtuel")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ➕ Ajouter une position")
        with st.form("add_position"):
            symbol_pf = st.text_input("Symbole", value="AAPL").upper()
            shares = st.number_input("Nombre d'actions", min_value=0.01, step=0.01)
            buy_price = st.number_input("Prix d'achat", min_value=0.01, step=0.01)
            
            if st.form_submit_button("Ajouter au portefeuille"):
                if symbol_pf and shares > 0:
                    if symbol_pf not in st.session_state.portfolio:
                        st.session_state.portfolio[symbol_pf] = []
                    
                    st.session_state.portfolio[symbol_pf].append({
                        'shares': shares,
                        'buy_price': buy_price,
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    st.success(f"✅ {shares} actions {symbol_pf} ajoutées")
    
    with col1:
        st.markdown("### 📊 Performance du portefeuille")
        
        if st.session_state.portfolio:
            portfolio_data = []
            total_value = 0
            total_cost = 0
            
            for symbol_pf, positions in st.session_state.portfolio.items():
                try:
                    ticker = yf.Ticker(symbol_pf)
                    current = ticker.history(period='1d')['Close'].iloc[-1]
                    
                    for pos in positions:
                        shares = pos['shares']
                        buy_price = pos['buy_price']
                        cost = shares * buy_price
                        value = shares * current
                        profit = value - cost
                        profit_pct = (profit / cost) * 100
                        
                        total_cost += cost
                        total_value += value
                        
                        portfolio_data.append({
                            'Symbole': symbol_pf,
                            'Actions': shares,
                            "Prix d'achat": f"${buy_price:.2f}",
                            'Prix actuel': f"${current:.2f}",
                            'Valeur': f"${value:,.2f}",
                            'Profit': f"${profit:,.2f}",
                            'Profit %': f"{profit_pct:.1f}%"
                        })
                except:
                    st.warning(f"Impossible de charger {symbol_pf}")
            
            # Métriques globales
            total_profit = total_value - total_cost
            total_profit_pct = (total_profit / total_cost) * 100 if total_cost > 0 else 0
            
            col1_1, col1_2, col1_3 = st.columns(3)
            col1_1.metric("Valeur totale", f"${total_value:,.2f}")
            col1_2.metric("Coût total", f"${total_cost:,.2f}")
            col1_3.metric(
                "Profit total",
                f"${total_profit:,.2f}",
                delta=f"{total_profit_pct:.1f}%"
            )
            
            # Tableau des positions
            st.markdown("### 📋 Positions détaillées")
            df_portfolio = pd.DataFrame(portfolio_data)
            st.dataframe(df_portfolio, use_container_width=True)
            
            # Graphique de répartition
            fig_pie = px.pie(
                names=[p['Symbole'] for p in portfolio_data],
                values=[float(p['Valeur'].replace('$', '').replace(',', '')) for p in portfolio_data],
                title="Répartition du portefeuille"
            )
            st.plotly_chart(fig_pie)
            
            # Bouton pour vider le portefeuille
            if st.button("🗑️ Vider le portefeuille"):
                st.session_state.portfolio = {}
                st.rerun()
        else:
            st.info("Aucune position dans le portefeuille. Ajoutez des actions pour commencer !")

# ============================================================================
# SECTION 3: ALERTES DE PRIX
# ============================================================================
elif menu == "🔔 Alertes de prix":
    st.subheader("🔔 Gestion des alertes de prix")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ➕ Créer une nouvelle alerte")
        with st.form("new_alert"):
            alert_symbol = st.text_input("Symbole", value=symbol).upper()
            alert_price = st.number_input("Prix cible ($)", min_value=0.01, step=0.01, value=float(current_price * 1.05))
            
            col_cond, col_type = st.columns(2)
            with col_cond:
                condition = st.selectbox("Condition", ["above", "below"])
            with col_type:
                alert_type = st.selectbox("Type", ["Permanent", "Une fois"])
            
            one_time = alert_type == "Une fois"
            
            if st.form_submit_button("Créer l'alerte"):
                st.session_state.price_alerts.append({
                    'symbol': alert_symbol,
                    'price': alert_price,
                    'condition': condition,
                    'one_time': one_time,
                    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                st.success(f"✅ Alerte créée pour {alert_symbol} à ${alert_price:.2f}")
    
    with col2:
        st.markdown("### 📋 Alertes actives")
        if st.session_state.price_alerts:
            for i, alert in enumerate(st.session_state.price_alerts):
                with st.container():
                    st.markdown(f"""
                    <div class='alert-box alert-warning'>
                        <b>{alert['symbol']}</b> - {alert['condition']} ${alert['price']:.2f}<br>
                        <small>Créée: {alert['created']} | {('Usage unique' if alert['one_time'] else 'Permanent')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Supprimer", key=f"del_alert_{i}"):
                        st.session_state.price_alerts.pop(i)
                        st.rerun()
        else:
            st.info("Aucune alerte active")

# ============================================================================
# SECTION 4: NOTIFICATIONS EMAIL
# ============================================================================
elif menu == "📧 Notifications email":
    st.subheader("📧 Configuration des notifications email")
    
    with st.form("email_config"):
        enabled = st.checkbox("Activer les notifications email", value=st.session_state.email_config['enabled'])
        
        col1, col2 = st.columns(2)
        with col1:
            smtp_server = st.text_input("Serveur SMTP", value=st.session_state.email_config['smtp_server'])
            smtp_port = st.number_input("Port SMTP", value=st.session_state.email_config['smtp_port'])
        
        with col2:
            email = st.text_input("Adresse email", value=st.session_state.email_config['email'])
            password = st.text_input("Mot de passe", type="password", value=st.session_state.email_config['password'])
        
        test_email = st.text_input("Email de test (optionnel)")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.form_submit_button("💾 Sauvegarder"):
                st.session_state.email_config = {
                    'enabled': enabled,
                    'smtp_server': smtp_server,
                    'smtp_port': smtp_port,
                    'email': email,
                    'password': password
                }
                st.success("Configuration sauvegardée !")
        
        with col_btn2:
            if st.form_submit_button("📨 Tester"):
                if test_email:
                    if send_email_alert(
                        "Test de notification",
                        "<h2>Ceci est un test</h2><p>Votre configuration email fonctionne correctement !</p>",
                        test_email
                    ):
                        st.success("Email de test envoyé !")
                    else:
                        st.error("Échec de l'envoi")
    
    # Aperçu de la configuration
    with st.expander("📋 Aperçu de la configuration"):
        st.json(st.session_state.email_config)

# ============================================================================
# SECTION 5: EXPORT DES DONNÉES
# ============================================================================
elif menu == "📤 Export des données":
    st.subheader("📤 Export des données")
    
    if hist is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Données historiques")
            st.dataframe(hist.tail(20))
            
            # Export CSV
            csv = hist.to_csv()
            st.download_button(
                label="📥 Télécharger en CSV",
                data=csv,
                file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### 📈 Rapport PDF")
            st.info("Génération de rapport PDF (simulée)")
            
            # Statistiques
            st.markdown("**Statistiques:**")
            stats = {
                'Moyenne': hist['Close'].mean(),
                'Écart-type': hist['Close'].std(),
                'Min': hist['Close'].min(),
                'Max': hist['Close'].max(),
                'Variation totale': f"{(hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100:.2f}%"
            }
            
            for key, value in stats.items():
                if isinstance(value, float):
                    st.write(f"{key}: ${value:.2f}")
                else:
                    st.write(f"{key}: {value}")
            
            # Export JSON
            json_data = {
                'symbol': symbol,
                'last_update': datetime.now().isoformat(),
                'current_price': current_price,
                'statistics': stats,
                'data': hist.reset_index().to_dict(orient='records')
            }
            
            st.download_button(
                label="📥 Télécharger en JSON",
                data=json.dumps(json_data, indent=2, default=str),
                file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.warning("Aucune donnée à exporter")

# ============================================================================
# SECTION 6: PRÉDICTIONS ML
# ============================================================================
elif menu == "🤖 Prédictions ML":
    st.subheader("🤖 Prédictions avec Machine Learning")
    
    if hist is not None and len(hist) > 30:
        st.markdown("### Modèle de prédiction (Régression polynomiale)")
        
        # Préparation des données
        df_pred = hist[['Close']].reset_index()
        df_pred['Days'] = (df_pred['Date'] - df_pred['Date'].min()).dt.days
        
        X = df_pred['Days'].values.reshape(-1, 1)
        y = df_pred['Close'].values
        
        # Configuration de la prédiction
        col1, col2 = st.columns(2)
        
        with col1:
            days_to_predict = st.slider("Jours à prédire", min_value=1, max_value=30, value=7)
            degree = st.slider("Degré du polynôme", min_value=1, max_value=5, value=2)
        
        with col2:
            st.markdown("### Options")
            show_confidence = st.checkbox("Afficher l'intervalle de confiance", value=True)
        
        # Entraînement du modèle
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        model.fit(X, y)
        
        # Prédictions
        last_day = X[-1][0]
        future_days = np.arange(last_day + 1, last_day + days_to_predict + 1).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        # Dates futures
        last_date = df_pred['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        # Visualisation
        fig_pred = go.Figure()
        
        # Données historiques
        fig_pred.add_trace(go.Scatter(
            x=df_pred['Date'],
            y=y,
            mode='lines',
            name='Historique',
            line=dict(color='blue')
        ))
        
        # Prédictions
        fig_pred.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Prédictions',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8)
        ))
        
        # Intervalle de confiance (simulé)
        if show_confidence:
            residuals = y - model.predict(X)
            std_residuals = np.std(residuals)
            
            upper_bound = predictions + 2 * std_residuals
            lower_bound = predictions - 2 * std_residuals
            
            fig_pred.add_trace(go.Scatter(
                x=future_dates + future_dates[::-1],
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='Intervalle de confiance (95%)'
            ))
        
        fig_pred.update_layout(
            title=f"Prédictions pour {symbol} - {days_to_predict} jours",
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Tableau des prédictions
        st.markdown("### 📋 Prédictions détaillées")
        pred_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Prix prédit': [f"${p:.2f}" for p in predictions],
            'Variation %': [f"{(p/current_price - 1)*100:.2f}%" for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True)
        
        # Métriques de performance
        st.markdown("### 📊 Performance du modèle")
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("RMSE", f"${rmse:.2f}")
        col_m2.metric("MAE", f"${mae:.2f}")
        col_m3.metric("R²", f"{model.score(X, y):.3f}")
        
        # Analyse des tendances
        st.markdown("### 📈 Analyse des tendances")
        last_price = current_price
        last_pred = predictions[-1]
        trend = "HAUSSIÈRE 📈" if last_pred > last_price else "BAISSIÈRE 📉" if last_pred < last_price else "NEUTRE ➡️"
        
        if last_pred > last_price * 1.05:
            strength = "Forte tendance haussière 🚀"
        elif last_pred > last_price:
            strength = "Légère tendance haussière 📈"
        elif last_pred < last_price * 0.95:
            strength = "Forte tendance baissière 🔻"
        elif last_pred < last_price:
            strength = "Légère tendance baissière 📉"
        else:
            strength = "Tendance latérale ⏸️"
        
        st.info(f"**Tendance prévue:** {trend} - {strength}")
        
    else:
        st.warning("Pas assez de données historiques pour faire des prédictions (minimum 30 points)")

# ============================================================================
# WATCHLIST ET DERNIÈRE MISE À JOUR
# ============================================================================
st.markdown("---")
col_w1, col_w2 = st.columns([3, 1])

with col_w1:
    st.subheader("📋 Watchlist")
    watchlist_cols = st.columns(len(st.session_state.watchlist))
    
    for i, sym in enumerate(st.session_state.watchlist):
        with watchlist_cols[i]:
            try:
                ticker = yf.Ticker(sym)
                price = ticker.history(period='1d')['Close'].iloc[-1]
                st.metric(sym, f"${price:.2f}")
            except:
                st.metric(sym, "N/A")

with col_w2:
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
    if auto_refresh:
        time.sleep(1)
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "📊 Stock Tracker Pro - Données fournies par yfinance | ⚠️ Données en temps réel avec délai possible"
    "</p>",
    unsafe_allow_html=True
)
