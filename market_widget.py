"""Provider-hosted quote widgets; never synthesize or cache quoted prices."""
import json

BENCHMARKS = {
    'Robusta': ('ICEEUR:RC1!', 'https://www.tradingview.com/symbols/ICEEUR-RC1!/', 'USD per metric tonne'),
    'Arabica': ('ICEUS:KC1!', 'https://www.tradingview.com/symbols/ICEUS-KC1!/', 'US cents per pound'),
}


def quote_widget_html(benchmark, theme='dark'):
    if benchmark not in BENCHMARKS or theme not in ('dark','light'):
        raise ValueError('Unsupported market widget configuration')
    symbol, url, _ = BENCHMARKS[benchmark]
    config=json.dumps({'symbol':symbol, 'width':'100%', 'locale':'en', 'colorTheme':theme, 'isTransparent':True})
    return f'''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0;background:transparent">
<div class="tradingview-widget-container">
<div class="tradingview-widget-container__widget"></div>
<div class="tradingview-widget-copyright"><a href="{url}" target="_blank" rel="noopener noreferrer"><span>{benchmark} futures by TradingView</span></a></div>
<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>{config}</script>
</div></body></html>'''


def render_quote_widgets(theme='dark'):
    import streamlit as st
    import streamlit.components.v1 as components
    columns=st.columns(2)
    for column, benchmark in zip(columns, BENCHMARKS):
        _, url, unit=BENCHMARKS[benchmark]
        with column:
            st.subheader(benchmark + ' futures')
            components.html(quote_widget_html(benchmark,theme),height=240,scrolling=False)
            st.caption(unit + ' · Continuous futures benchmark, not a local farm-gate offer.')
            st.link_button('Open '+benchmark+' market quote', url)
    st.caption('Source: TradingView / ICE futures. Provider-hosted panels update while open; exchange data may be delayed or unavailable by region or entitlement. Check the provider’s quote time and delay indicator. These are not guaranteed real-time prices.')
