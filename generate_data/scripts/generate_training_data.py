from ast import Dict
from openai import OpenAI
import os
from tqdm import tqdm
import json
from src.logger import get_console_logger
from src.paths import DATA_DIR






logger = get_console_logger()

MODEL = "gpt-3.5-turbo-instruct"

PROMPT_TEMPLATE = """
You are an expert in the stock and crypto markets. I will give you some information about myself and you will provide me with good investment advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

Please provide concrete advice in less than 100 tokens, and justify your answer based on the news provided in the context.
"""

EXAMPLES = [
    {
        "about_me": "I'm a 40 year old engineer. I've saved up a considerable amount and want to invest for early retirement. What's the best strategy?",
        "context": "Early retirement goals. Market volatility due to global events. Give a response that takes into consideration macroeconomic factors to reduce risk, using the idea of risk parity, which is the balancing risks by diversifying across asset classes that are not correlated."
    },
    {
        "about_me": "I'm a 55 year old marketing executive. I have a 401(k) but haven't reviewed it in years. How can I optimize my retirement savings?",
        "context": "Neglected 401(k) account. The Federal Reserve's monetary policy may affect investments. Give a response that takes into consideration macroeconomic factors to reduce risk, using the idea of risk parity, which is the balancing risks by diversifying across asset classes that are not correlated."
    },
    {
        "about_me": "I'm a 25 year old entrepreneur. I have a high-risk tolerance. Where should I invest to maximize returns?",
        "context": "High-risk tolerance. Bullish market sentiment. Give a response that considers potential high returns but emphasizes risk management."
    },
    {
        "about_me": "I'm a 60 year old retiree. I've heard about annuities. Are they a safe option for generating retirement income?",
        "context": "Interest in annuities as a retirement income source. Concerns about financial stability. Give a response that discusses annuities as an option with pros and cons."
    },
    {
        "about_me": "I'm a 35 year old sales manager. I want to invest in stocks but don't have time to research individual companies. What's a good approach?",
        "context": "Limited time for stock research. Bullish stock market trends. Give a response that suggests diversified index funds or ETFs."
    },
    {
        "about_me": "I'm a 45 year old parent. I want to save for my child's future education. What's the best way to plan for college expenses?",
        "context": "Long-term college savings goals. Rising education costs. Give a response that recommends college savings accounts and diversified investments."
    },
    {
        "about_me": "I'm a 28 year old newlywed. My spouse and I want to invest together. How can we build a joint investment strategy?",
        "context": "Joint investment goals for newlyweds. Economic uncertainty. Give a response that discusses joint investment planning and risk management."
    },
    {
        "about_me": "I'm a 50 year old manager. I'm concerned about a potential market crash. How can I protect my investments?",
        "context": "Fear of market downturn. Historical market crashes. Give a response that advises diversification, asset allocation, and risk mitigation strategies."
    },
    {
        "about_me": "I'm a 30 year old tech entrepreneur. I want to invest my company's excess cash. Where should I park our funds for growth?",
        "context": "Company cash reserves. Interest in maximizing returns. Give a response that suggests corporate investment options and risk management."
    },
    {
        "about_me": "I'm a 22 year old recent graduate. I have student loans to pay off. Should I start investing or focus on debt repayment?",
        "context": "Student loan debt vs. investing dilemma. Current low-interest rates. Give a response that discusses debt repayment strategies and gradual investing."
    },
    {
        "about_me": "I'm a 48 year old real estate agent. Is it wise to invest in real estate given the current market conditions?",
        "context": "Real estate professional's interest in real estate investments. Housing market dynamics. Give a response that analyzes the real estate market's pros and cons."
    },
    {
        "about_me": "I'm a 29 year old teacher. I have a stable job but limited savings. How can I start building wealth on a teacher's salary?",
        "context": "Low savings on a teacher's income. Interest in wealth building. Give a response that suggests budgeting, saving, and investment options."
    },
    {
        "about_me": "I'm a 37 year old consultant. I have variable income. How can I create a stable financial future?",
        "context": "Irregular income from consulting work. Uncertainty in income streams. Give a response that emphasizes budgeting, emergency funds, and stable investments."
    },
    {
        "about_me": "I'm a 65 year old retiree. I want to leave a financial legacy for my grandchildren. What's the best way to plan for generational wealth?",
        "context": "Generational wealth planning. Desire to secure grandchildren's financial future."
	},
	{
        "about_me": "I'm a 26 year old software developer. I want to start investing, but I'm concerned about environmental and social impact. How can I invest ethically?",
        "context": "Interest in socially responsible investing. Growing focus on ESG (Environmental, Social, Governance) investments. Give a response that explores sustainable investment options."
    },
    {
        "about_me": "I'm a 55 year old business owner. My business is thriving, and I want to invest my profits. What are the tax implications of business investments?",
        "context": "Successful business with surplus funds. Tax considerations for business investments. Give a response that touches on tax-efficient investment strategies."
    },
    {
        "about_me": "I'm a 31 year old pharmacist. I have a sizeable student loan debt. Should I prioritize paying off my loans or investing for the future?",
        "context": "Student loan debt vs. investment decision. Interest rate on student loans. Give a response that discusses the balance between debt repayment and investing."
    },
    {
        "about_me": "I'm a 58 year old attorney. I'm nearing retirement and want to ensure a comfortable retirement lifestyle. How can I safeguard my financial future?",
        "context": "Approaching retirement age. Desire for financial security. Give a response that covers retirement planning, income streams, and risk management."
    },
    {
        "about_me": "I'm a 42 year old chef. I've received an unexpected windfall. What's the best way to invest this unexpected money?",
        "context": "Receiving a financial windfall. Interest in responsible wealth management. Give a response that discusses windfall investment strategies and risk mitigation."
    },
    {
        "about_me": "I'm a 34 year old engineer. I have a strong belief in the potential of renewable energy. How can I invest in the green energy sector?",
        "context": "Passion for renewable energy investments. Growth of the green energy industry. Give a response that explores investment opportunities in clean energy."
    },
    {
        "about_me": "I'm a 48 year old marketing director. I've heard about robo-advisors. Are they a good option for managing my investments?",
        "context": "Interest in robo-advisors as an investment tool. Busy work schedule. Give a response that explains the benefits and limitations of robo-advisors."
    },
    {
        "about_me": "I'm a 30 year old graphic designer. I have a side hustle and want to invest my extra income. How can I grow my side hustle earnings?",
        "context": "Side hustle income with investment intentions. Desire to maximize side income. Give a response that suggests investment options for side hustle earnings."
    },
    {
        "about_me": "I'm a 56 year old sales manager. I've been investing in stocks for years. Should I diversify my portfolio to reduce risk?",
        "context": "Experienced stock market investor. Concerns about portfolio risk. Give a response that discusses portfolio diversification strategies."
    },
    {
        "about_me": "I'm a 23 year old recent graduate. I have no savings, but I want to start investing for my future. What's the first step?",
        "context": "Starting to invest with limited savings. Long-term financial goals. Give a response that outlines the first steps to begin investing."
    },
    {
        "about_me": "I'm a 44 year old dentist. I have a substantial amount saved, but I'm worried about inflation eroding my wealth. What should I do?",
        "context": "Concerns about the impact of inflation on savings. Desire to protect wealth. Give a response that suggests inflation-hedging investment options."
    },
    {
        "about_me": "I'm a 32 year old marketing manager. I want to invest in stocks, but I'm not sure how to analyze companies. How can I evaluate stocks?",
        "context": "Interest in stock market investing. Lack of stock analysis knowledge. Give a response that provides basic guidelines for evaluating stocks."
    },
    {
        "about_me": "I'm a 50 year old teacher. I have a pension, but I want to supplement my retirement income. What are my investment options?",
        "context": "Teacher with a pension. Desire for additional retirement income. Give a response that explores retirement income strategies and investment choices."
    },
    {
        "about_me": "I'm a 27 year old marketing coordinator. I've recently inherited a substantial sum of money. How can I manage this inheritance wisely?",
        "context": "Inheriting a significant sum of money. Responsibility to manage inherited wealth. Give a response that discusses inheritance management and long-term planning."
    },
    {
        "about_me": "I'm a 35 year old engineer. I'm interested in day trading, but I'm aware of its risks. Should I pursue day trading as an investment strategy?",
        "context": "Interest in day trading. Awareness of day trading risks. Give a response that provides insights into day trading and risk management."
    },
    {
        "about_me": "I'm a 51 year old nurse. I have a 401(k) but am unsure about my investment choices. What's the best way to allocate my 401(k) funds?",
        "context": "401(k) account holder. Need for asset allocation guidance. Give a response that suggests 401(k) asset allocation strategies."
    },
    {
        "about_me": "I'm a 29 year old software developer. I've been hearing about cryptocurrency. Is it a good time to invest in cryptocurrencies?",
        "context": "Curiosity about cryptocurrency investments. Volatile cryptocurrency market. Give a response that explores the cryptocurrency market's pros and cons."
    },
    {
        "about_me": "I'm a 46 year old small business owner. I want to invest some of my business profits. What are the tax implications of business investments?",
        "context": "Small business owner with investment goals. Tax considerations for business investments. Give a response that addresses tax-efficient investment strategies."
    },
    {
        "about_me": "I'm a 33 year old marketing specialist. I've received stock options from my company. How can I make the most of these stock options?",
        "context": "Employee with stock options. Desire to optimize stock options. Give a response that explains stock options and strategies for maximizing their value."
    },
    {
        "about_me": "I'm a 63 year old retiree. I have a significant amount in my retirement accounts. How can I ensure a reliable income during retirement?",
        "context": "Retiree with substantial retirement savings. Need for dependable retirement income. Give a response that covers retirement income strategies and risk management."
    },
    {
        "about_me": "I'm a 38 year old pharmacist. I'm concerned about healthcare costs in retirement. How can I prepare financially for healthcare expenses?",
        "context": "Preparation for healthcare expenses in retirement. Rising healthcare costs. Give a response that explores healthcare cost planning and investment strategies."
    },
    {
        "about_me": "I'm a 24 year old student with some part-time income. How can I start building wealth while still in school?",
        "context": "Student interested in wealth building. Limited income. Give a response that suggests budgeting, saving, and investment options for students."
    },
    {
        "about_me": "I'm a 49 year old sales director. I want to invest for my child's future. What's the best way to save for their education?",
        "context": "Desire to save for a child's education. College tuition planning. Give a response that discusses college savings accounts and investment strategies for education."
    },
    {
        "about_me": "I'm a 28 year old teacher. I've heard about 529 plans. Are they a good option for saving for my child's education?",
        "context": "Interest in 529 college savings plans. Desire to save for a child's education. Give a response that explains 529 plans and their benefits."
    },
    {
        "about_me": "I'm a 45 year old real estate agent. I've invested in residential properties, but I'm considering commercial real estate. Is it a wise move?",
        "context": "Real estate investor exploring commercial properties. Real estate market dynamics. Give a response that analyzes the pros and cons of commercial real estate investments."
    },
    {
        "about_me": "I'm a 30 year old freelance writer. My income fluctuates. How can I save and invest during periods of irregular income?",
        "context": "Freelancer with income variability. Financial stability during income fluctuations. Give a response that emphasizes budgeting, emergency funds, and investing strategies for freelancers."
    },
    {
        "about_me": "I'm a 55 year old artist. I have a substantial art collection. Should I consider art as an investment?",
        "context": "Art collector considering art as an investment. Art market trends. Give a response that discusses art as an investment asset class."
    },
    {
        "about_me": "I'm a 33 year old marketing manager. I want to invest with a focus on environmental sustainability. What are my options for green investments?",
        "context": "Interest in environmentally sustainable investments. Growth of green investment opportunities. Give a response that explores environmentally responsible investment options."
    },
    {
        "about_me": "I'm a 50 year old accountant. I have some extra funds and want to explore international investments. How can I diversify internationally?",
        "context": "Interest in diversifying with international investments. Global investment opportunities. Give a response that discusses international diversification strategies."
    },
    {
        "about_me": "I'm a 29 year old healthcare professional. I've been approached by a financial advisor. Should I consider professional financial advice?",
        "context": "Financial advisor recommendation. Desire for expert financial guidance. Give a response that outlines the benefits of professional financial advice and considerations when choosing an advisor."
    },
    {
        "about_me": "I'm a 56 year old retiree. I want to travel during retirement. How can I allocate my investments to fund my travel plans?",
        "context": "Retiree with travel goals. Balancing financial needs and travel aspirations. Give a response that discusses investment allocation for retirement travel."
    },
    {
        "about_me": "I'm a 37 year old IT professional. I have some savings and want to invest for my children's future. What's the best strategy for their financial security?",
        "context": "Desire to secure children's financial future. Long-term financial planning. Give a response that suggests investment strategies for children's financial security."
    },
    {
        "about_me": "I'm a 62 year old retiree. I have a mortgage on my home. Should I use my retirement savings to pay it off or keep investing?",
        "context": "Debating mortgage repayment vs. investment. Retirement income considerations. Give a response that explores the pros and cons of paying off the mortgage versus continued investing."
    },
    {
        "about_me": "I'm a 31 year old entrepreneur. I've built a successful business. How can I invest my business profits for long-term growth?",
        "context": "Successful business owner with investment goals. Business profit reinvestment strategies. Give a response that discusses business profit investment options for growth."
    },
    {
        "about_me": "I'm a 47 year old engineer. I've been saving for my child's wedding. How can I ensure I have enough funds for their dream wedding?",
        "context": "Saving for a child's wedding. Wedding cost planning. Give a response that explores wedding savings and investment strategies."
    },
    {
        "about_me": "I'm a 30 year old teacher. I want to build an emergency fund. What's the best way to start saving for unexpected expenses?",
        "context": "Desire to establish an emergency fund. Importance of financial safety nets. Give a response that outlines the steps to build an emergency fund."
    },
    {
        "about_me": "I'm a 58 year old retiree. I have a comfortable retirement income but want to leave a financial legacy for charity. How can I plan for charitable giving?",
        "context": "Charitable giving intentions in retirement. Philanthropic legacy planning. Give a response that discusses charitable giving strategies and tax considerations."
    },
    {
        "about_me": "I'm a 25 year old recent graduate. I have student loans to pay off. Should I prioritize paying off my loans or start investing?",
        "context": "Balancing student loan debt and investing goals. Interest in financial stability. Give a response that explores the balance between student loan repayment and investing."
    },
    {
        "about_me": "I'm a 52 year old manager. I want to invest in real estate but don't have the time to manage properties. What are my options for passive real estate investments?",
        "context": "Interest in passive real estate investments. Real estate market dynamics. Give a response that explores passive real estate investment options."
    },
    {
        "about_me": "I'm a 36 year old sales executive. I want to build a diverse investment portfolio. What are some alternative assets I can consider?",
        "context": "Interest in diversified investment portfolios. Exploration of alternative investment options. Give a response that discusses alternative assets like cryptocurrencies, precious metals, and more."
    },
    {
        "about_me": "I'm a 64 year old retiree. I have a fixed income from my pension. How can I invest to supplement my retirement income?",
        "context": "Fixed retirement income. Desire for supplemental income. Give a response that explores investment strategies to supplement pension income."
    },
    {
        "about_me": "I'm a 39 year old marketing manager. I have a mortgage on my home. Should I refinance my mortgage to free up cash for investments?",
        "context": "Mortgage refinancing considerations for investment. Interest rate environment. Give a response that discusses the benefits and risks of mortgage refinancing for investment purposes"
	},

    {
        "about_me": "I'm a 40 year old engineer. I've been saving for retirement, but I'm worried about market crashes. How can I protect my retirement savings?",
        "context": "Concerns about market volatility and retirement savings. Strategies to safeguard investments during market downturns."
    },
    {
        "about_me": "I'm a 26 year old entrepreneur. I have some extra cash, but I'm unsure about investing. How can I get started with investing?",
        "context": "Interest in investing with available cash. Investment basics and getting started."
    },
    {
        "about_me": "I'm a 58 year old business owner. I have significant business profits. What's the best way to invest my business earnings for growth?",
        "context": "Successful business owner seeking investment options for business profits. Strategies to maximize returns."
    },
    {
        "about_me": "I'm a 32 year old software developer. I want to diversify my investment portfolio. What are some alternative investments to consider?",
        "context": "Interest in diversifying investments. Exploring alternative investment opportunities such as real estate, cryptocurrencies, and more."
    },
    {
        "about_me": "I'm a 51 year old nurse. I have a 401(k) plan, but I'm unsure if I'm making the right investment choices. How can I optimize my retirement portfolio?",
        "context": "Desire to optimize 401(k) investments. Strategies to make informed choices for retirement portfolio."
    },
    {
        "about_me": "I'm a 29 year old tech enthusiast. I want to invest in tech stocks, but I'm concerned about their valuations. Should I wait for a tech market correction?",
        "context": "Interest in tech stock investments. Evaluation of tech market conditions and timing considerations."
    },
    {
        "about_me": "I'm a 48 year old financial analyst. I want to explore international investments. What's the best approach to diversify my portfolio globally?",
        "context": "Interest in global diversification. Strategies and considerations for international investments."
    },
    {
        "about_me": "I'm a 34 year old artist with variable income. How can I build a stable financial future despite irregular earnings?",
        "context": "Income fluctuations as an artist. Financial stability and savings strategies for variable income."
    },
    {
        "about_me": "I'm a 61 year old retiree living on a fixed pension. How can I generate additional income during retirement?",
        "context": "Fixed retirement income. Investment options to supplement pension income."
    },
    {
        "about_me": "I'm a 35 year old lawyer. I have some savings and want to invest ethically. What are my options for sustainable investments?",
        "context": "Interest in ethical and sustainable investments. Exploring environmentally responsible investment opportunities."
    },
    {
        "about_me": "I'm a 49 year old marketing manager. I want to invest for my child's future education. What's the best way to save for their college fund?",
        "context": "Desire to save for a child's education. Strategies for college fund planning and investments."
    },
    {
        "about_me": "I'm a 27 year old recent graduate. I have student loans and want to start investing. How can I manage both priorities?",
        "context": "Balancing student loan debt and investing goals. Strategies for financial stability and growth."
    },
    {
        "about_me": "I'm a 53 year old real estate investor. I own residential properties, but I'm considering commercial real estate. Is it a wise move?",
        "context": "Real estate investor exploring commercial properties. Analysis of commercial real estate investments."
    },
    {
        "about_me": "I'm a 36 year old sales executive. I've received a large bonus. How can I invest this windfall wisely?",
        "context": "Investment options for a financial windfall. Strategies to make the most of a bonus."
    },
    {
        "about_me": "I'm a 59 year old retiree. I want to leave an inheritance for my children. How can I plan for generational wealth?",
        "context": "Desire to leave an inheritance. Estate planning and generational wealth strategies."
    },
    {
        "about_me": "I'm a 42 year old pharmacist. I'm concerned about healthcare costs in retirement. How can I prepare financially for healthcare expenses?",
        "context": "Preparation for healthcare expenses in retirement. Rising healthcare costs. Healthcare cost planning and investment strategies."
    },
    {
        "about_me": "I'm a 24 year old student with some part-time income. How can I start building wealth while still in school?",
        "context": "Student interested in wealth building. Limited income. Budgeting, saving, and investment options for students."
    },
    {
        "about_me": "I'm a 49 year old sales director. I want to invest for my child's future. What's the best way to save for their education?",
        "context": "Desire to save for a child's education. College tuition planning. College savings accounts and investment strategies."
    },
    {
        "about_me": "I'm a 28 year old teacher. I've heard about 529 plans. Are they a good option for saving for my child's education?",
        "context": "Interest in 529 college savings plans. Desire to save for a child's education. Explaining 529 plans and their benefits."
    },
    {
        "about_me": "I'm a 45 year old real estate agent. I've invested in residential properties, but I'm considering commercial real estate. Is it a wise move?",
        "context": "Real estate investor exploring commercial properties. Real estate market dynamics and the pros and cons of commercial real estate investments."
    },
    {
        "about_me": "I'm a 30 year old freelance writer. My income fluctuates. How can I save and invest during periods of irregular income?",
        "context": "Freelancer with income variability. Financial stability during income fluctuations. Budgeting, emergency funds, and investing strategies for freelancers."
    },
    {
        "about_me": "I'm a 55 year old artist. I have a substantial art collection. Should I consider art as an investment?",
        "context": "Art collector considering art as an investment. Art market trends and the potential of art as an investment asset class."
    },
    {
        "about_me": "I'm a 33 year old marketing manager. I want to invest with a focus on environmental sustainability. What are my options for green investments?",
        "context": "Interest in environmentally sustainable investments. Growth of green investment opportunities and options for environmentally responsible investing."
    },
    {
        "about_me": "I'm a 50 year old accountant. I have some extra funds and want to explore international investments. How can I diversify internationally?",
        "context": "Interest in diversifying with international investments. Global investment opportunities and strategies for international diversification."
    },
    {
        "about_me": "I'm a 29 year old healthcare professional. I've been approached by a financial advisor. Should I consider their investment recommendations?",
        "context": "Healthcare professional considering financial advice. Evaluating financial advisors and their investment recommendations."
    },
    {
        "about_me": "I'm a 47 year old small business owner. I want to invest my business profits. What are the best investment options for entrepreneurs?",
        "context": "Small business owner seeking investment options for business profits. Investment strategies for entrepreneurs and business owners."
    },
    {
        "about_me": "I'm a 37 year old engineer. I have a 401(k) plan, but I'm unsure about my investment choices. How can I make informed decisions for my retirement account?",
        "context": "Desire to make informed 401(k) investment choices. Strategies for optimizing retirement account investments."
    },
    {
        "about_me": "I'm a 25 year old recent graduate. I've just started my first job. How can I begin investing for my financial future?",
        "context": "New graduate entering the workforce. Basics of investing for beginners and starting a financial future."
    },
    {
        "about_me": "I'm a 63 year old retiree. I want to ensure my retirement savings last throughout retirement. How can I manage my finances for longevity?",
        "context": "Retiree concerned about financial longevity. Strategies for managing retirement finances and ensuring long-term sustainability."
    },
    {
        "about_me": "I'm a 31 year old tech entrepreneur. I have a substantial net worth. How can I protect and grow my wealth in a changing economic landscape?",
        "context": "High-net-worth individual seeking wealth protection and growth strategies. Navigating economic changes and investment opportunities."
    },
    {
        "about_me": "I'm a 52 year old teacher. I want to invest for retirement. What's the best investment strategy for long-term growth and stability?",
        "context": "Desire to invest for retirement. Long-term investment strategies and options for retirement planning."
    },
    {
        "about_me": "I'm a 38 year old IT professional. I have some savings and want to invest. What are the key factors to consider before making investment decisions?",
        "context": "Interest in investing with available savings. Key factors and considerations for informed investment decisions."
    },
    {
        "about_me": "I'm a 22 year old recent immigrant. I want to build a financial future in a new country. How can I navigate the financial system and make wise investments?",
        "context": "New immigrant navigating financial systems in a new country. Financial planning and investment guidance for newcomers."
    },
    {
        "about_me": "I'm a 44 year old financial consultant. I want to explore impact investing. What are the opportunities for investing with a social and environmental focus?",
        "context": "Interest in impact investing. Opportunities and strategies for socially and environmentally responsible investments."
    },
    {
        "about_me": "I'm a 56 year old small business owner. I want to exit my business and invest the proceeds. What are the best exit and investment strategies?",
        "context": "Small business owner planning to exit the business. Exit strategies and investment options for business proceeds."
    },
    {
        "about_me": "I'm a 43 year old nurse. I have a 403(b) plan. How can I make the most of my retirement plan and achieve my financial goals?",
        "context": "Desire to optimize 403(b) retirement plan. Strategies for achieving financial goals through retirement planning."
    },
    {
        "about_me": "I'm a 26 year old marketing professional. I want to start investing, but I'm risk-averse. What are the safest investment options?",
        "context": "Risk-averse individual looking for safe investment options. Exploring low-risk investment opportunities."
    },
    {
        "about_me": "I'm a 68 year old retiree. I want to explore dividend stocks for regular income. How can I build a dividend-focused portfolio?",
        "context": "Retiree seeking regular income from dividends. Building a dividend-focused investment portfolio."
    },
    {
        "about_me": "I'm a 29 year old scientist. I want to invest in biotech companies. How can I evaluate and select promising biotech stocks?",
        "context": "Interest in investing in biotech stocks. Strategies for evaluating and selecting promising biotechnology companies."
    },
    {
        "about_me": "I'm a 49 year old pharmacist. I have a significant amount of savings. What's the best approach to grow my wealth for retirement?",
        "context": "Desire to grow wealth for retirement. Investment strategies and approaches for long-term wealth accumulation."
    },
    {
        "about_me": "I'm a 35 year old real estate agent. I want to invest in rental properties. What are the key considerations for successful real estate investing?",
        "context": "Interest in rental property investments. Key considerations and strategies for successful real estate investing."
    },
    {
        "about_me": "I'm a 32 year old lawyer. I have a stable job but want to invest in startups. How can I get started with angel investing?",
        "context": "Interest in angel investing. Getting started with startup investments and angel investing strategies."
    },
    {
        "about_me": "I'm a 60 year old retiree. I want to diversify my investment portfolio. What are some alternative investment options for diversification?",
        "context": "Desire to diversify investments. Exploring alternative investment options for portfolio diversification."
    },
    {
        "about_me": "I'm a 30 year old graphic designer. I want to invest in art. How can I start building an art collection as an investment?",
        "context": "Interest in art investment. Strategies for building an art collection for investment purposes."
    },
    {
        "about_me": "I'm a 54 year old financial advisor. I want to provide my clients with sustainable investment options. What are the best ESG investment opportunities?",
        "context": "Interest in offering ESG investment options to clients. Identifying the best environmental, social, and governance (ESG) investment opportunities."
    },
    {
        "about_me": "I'm a 33 year old educator. I want to invest in my retirement fund. How can I make the most of my 403(b) plan?",
        "context": "Desire to maximize a 403(b) retirement plan. Strategies for optimizing retirement savings through 403(b) plans."
    },
    {
        "about_me": "I'm a 24 year old recent graduate with student loans. Should I prioritize paying off my debt or start investing?",
        "context": "Balancing student loan debt and investing goals. Strategies for managing student loans while building wealth."
    },
    {
        "about_me": "I'm a 41 year old engineer. I want to invest in emerging markets. What are the opportunities and risks of investing in developing economies?",
        "context": "Interest in emerging market investments. Evaluating opportunities and risks in developing economies."
    },
    {
        "about_me": "I'm a 37 year old sales manager. I want to invest in tech startups. How can I identify promising tech companies for investment?",
        "context": "Interest in tech startup investments. Strategies for identifying and investing in promising technology"
    }
]

# TODO: CREATE DOTFILE WITH OPENAI KEY
OpenAI.api_key = os.environ["OPENAI_API_KEY"]

def build_prompt(example: Dict) -> str:
    return PROMPT_TEMPLATE.format(
        ABOUT_ME = example["about_me"],
        CONTEXT = example["context"], 
    )

def run():
    output = []
    client = OpenAI()
    for example in tqdm(EXAMPLES):
        prompt = build_prompt(example)
        logger.info(f"{prompt=}")

        res = client.completions.create(
            model=MODEL,
            prompt=prompt,
            temperature=0,
            max_tokens=100,
        )
        

        res = res.choices[0].text
        logger.info(f"{res}=")
        
        output.append({**example, "response": res})
    
    with open(DATA_DIR / "training_data.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    run()

    


        