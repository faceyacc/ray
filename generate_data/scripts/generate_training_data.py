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
You are an expert in the stock and crypto markets. I will give you some information about myself, a question, and you will provide me with good investment advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

# QUESTION
{QUESTION}

Please provide concrete advice in less than 100 tokens, and justify your answer based on the news provided in the context.
"""

EXAMPLES = example_prompts = [
    {"about_me": "I'm a 30-year-old teacher. I've been saving for a while and want to ensure financial stability for my family.",
     "context": "Looking for long-term financial security. Concerned about inflation and economic downturns. Interested in a mix of stocks and bonds.",
     "question": "What strategies can I use to hedge against inflation while ensuring steady growth for my savings?"},

    {"about_me": "I'm a 25-year-old software developer. Recently started my career and want to maximize my earnings.",
     "context": "High earning potential at a young age. Curious about tech stocks and emerging markets. Willing to take on more risk for higher returns.",
     "question": "Should I focus on aggressive growth stocks or diversify into international markets for long-term gains?"},

    {"about_me": "I'm a 50-year-old business owner. Planning for retirement and thinking about my financial legacy.",
     "context": "Nearing retirement with substantial savings. Interested in low-risk, income-generating investments. Considering tax implications.",
     "question": "What are the best options for tax-efficient, low-risk investments that provide a steady income?"},

    {"about_me": "I'm a 35-year-old freelance artist. My income varies, so I need a flexible and secure investment plan.",
     "context": "Irregular income but a need for financial security. Looking for investments that offer liquidity and minimal risk.",
     "question": "What are the safest investment options that offer quick liquidity for someone with an unpredictable income?"},

    {"about_me": "I'm a 28-year-old nurse. I've saved a modest amount and am ready to start investing.",
     "context": "New to investing with a stable job. Seeking straightforward, low-risk investment options. Interested in long-term growth.",
     "question": "As a beginner in investing, what are some simple and low-risk options that I can start with?"},

    {"about_me": "I'm a 45-year-old lawyer. I've accumulated wealth and am looking to diversify my portfolio.",
     "context": "Mid-career with significant savings. Exploring international and alternative investment opportunities. Risk tolerance is moderate.",
     "question": "How can I effectively diversify my portfolio to include international and alternative investments?"},

    {"about_me": "I'm a 38-year-old marketing executive. I'm passionate about sustainable and ethical investing.",
     "context": "Interested in aligning investments with personal values. Focusing on environmental, social, and governance (ESG) criteria.",
     "question": "What are the most promising opportunities in sustainable and ethical investing right now?"},

    {"about_me": "I'm a 22-year-old recent college graduate. Just starting my career and want to build wealth.",
     "context": "Early in career with potential for growth. Keen on learning about stock market investing and retirement funds.",
     "question": "What are the best investment strategies for someone just starting their career and looking to grow their wealth?"},

    {"about_me": "I'm a 55-year-old government employee. Retirement is on the horizon, and I want to protect my nest egg.",
     "context": "Close to retirement, focusing on capital preservation. Interested in low-volatility investments and stable income sources.",
     "question": "What are the safest investment strategies for someone nearing retirement to protect their savings?"},

    {"about_me": "I'm a 32-year-old entrepreneur. My business is doing well, and I'd like to invest my profits wisely.",
     "context": "Reinvesting business profits for personal wealth. Looking for a balance of growth and stability in investments.",
     "question": "How should I approach investing my business profits for personal financial growth?"},
	{"about_me": "I'm a 42-year-old architect. I want to start investing in more than just my 401(k).",
     "context": "Looking to expand beyond traditional retirement accounts. Interested in real estate and tech sector investments.",
     "question": "What are some effective ways to diversify my investments outside of my 401(k) plan?"},

    {"about_me": "I'm a 26-year-old digital marketer. I want to take advantage of my youth to build a strong financial foundation.",
     "context": "Early in career, willing to take calculated risks. Interested in a mix of stocks, ETFs, and possibly starting a small side business.",
     "question": "How can I balance risk and reward in my investment strategy at this early stage of my career?"},

    {"about_me": "I'm a 48-year-old chef and restaurant owner. I'm looking to invest my savings in something other than my business.",
     "context": "Desire to diversify investments outside of the restaurant industry. Interested in low-maintenance, long-term investment options.",
     "question": "What are some passive investment opportunities that would be suitable for someone in my situation?"},

    {"about_me": "I'm a 35-year-old human resources manager. I'm cautious about investing but want to start building a portfolio.",
     "context": "Risk-averse but recognizes the need to invest. Looking for safe, stable investment options with decent returns.",
     "question": "What are the best investment options for someone who is risk-averse but wants to start building wealth?"},

    {"about_me": "I'm a 29-year-old graphic designer. I want to start saving for a home and possibly freelance full-time.",
     "context": "Saving for a major purchase while considering a career change. Needs flexibility and stability in investments.",
     "question": "How should I approach my investment strategy to save for a home while keeping my options open for freelancing?"},

    {"about_me": "I'm a 53-year-old teacher approaching retirement. I'm looking for ways to supplement my pension.",
     "context": "Close to retirement with a pension plan in place. Interested in additional income sources post-retirement.",
     "question": "What are some reliable investment options that can provide supplemental income during retirement?"},

    {"about_me": "I'm a 31-year-old pharmacist. I have a stable job and want to start investing wisely.",
     "context": "Steady income with a focus on long-term financial stability. Interested in a balanced portfolio with a mix of stocks and bonds.",
     "question": "What would be a balanced investment approach for someone with a stable job and a long-term outlook?"},

    {"about_me": "I'm a 40-year-old freelance photographer. My income is variable, so I need a flexible investment plan.",
     "context": "Irregular income streams but a keen interest in building wealth. Looking for investments with potential for high returns.",
     "question": "What are some investment strategies that offer high growth potential and flexibility for someone with an irregular income?"},

    {"about_me": "I'm a 27-year-old engineer. I'm interested in sustainable and tech-focused investments.",
     "context": "Passionate about technology and sustainability. Looking to invest in renewable energy and innovative tech startups.",
     "question": "What are the emerging trends in sustainable and tech investments that I should consider?"},

    {"about_me": "I'm a 60-year-old retiree. I want to manage my savings to ensure they last throughout my retirement.",
     "context": "Already retired and focused on preserving wealth. Interested in low-risk, income-generating investments.",
     "question": "How can I best manage my savings to ensure a steady income throughout my retirement years?"},

    {"about_me": "I'm a 34-year-old school principal. I want to start planning for my children's education and my retirement.",
     "context": "Balancing saving for kids' education and own retirement. Interested in education savings plans and retirement accounts.",
     "question": "How can I effectively save for my children's education while also ensuring my retirement is secure?"},

    {"about_me": "I'm a 29-year-old professional athlete. My career earnings are high but might be short-lived.",
     "context": "High current income with potentially limited career span. Looking for ways to secure financial stability post-career.",
     "question": "What are the best investment strategies for someone with a high but potentially short-term income?"},

    {"about_me": "I'm a 45-year-old journalist. I've been focusing on my career and neglected my financial planning.",
     "context": "Mid-career with a moderate amount of savings. Interested in catching up on retirement savings and investing in the stock market.",
     "question": "As a mid-career professional who hasn't focused much on investing, what strategies should I use to catch up?"},

    {"about_me": "I'm a 38-year-old IT consultant. I'm interested in investing in technology and innovation.",
     "context": "Tech-savvy with a keen interest in the tech industry. Looking to invest in high-growth tech companies and startups.",
     "question": "What are some promising areas within the tech sector that offer good investment opportunities?"},

    {"about_me": "I'm a 50-year-old small business owner. I want to ensure I have a diverse investment portfolio.",
     "context": "Established business owner looking to diversify. Interested in exploring international markets and alternative assets.",
     "question": "How can I diversify my investment portfolio to include international and alternative assets effectively?"},

    {"about_me": "I'm a 23-year-old recent graduate. I'm overwhelmed with student loans but want to start investing.",
     "context": "Dealing with student debt but eager to start investing. Looking for low-risk, low-capital investment options.",
     "question": "How can I balance paying off student loans with beginning to invest for my future?"},

    {"about_me": "I'm a 55-year-old healthcare professional. I want to invest in healthcare and pharmaceutical sectors.",
     "context": "Industry knowledge in healthcare. Interested in investing in healthcare stocks, biotech, and pharmaceutical companies.",
     "question": "What should I consider when investing in the healthcare and pharmaceutical sectors given my professional background?"},

    {"about_me": "I'm a 47-year-old government official. I'm looking for secure investment options for my retirement.",
     "context": "Stable job but cautious about investments. Seeking low-risk, steady growth investments for retirement.",
     "question": "What are the safest and most reliable investment options for a government employee nearing retirement?"},

    {"about_me": "I'm a 26-year-old entrepreneur. I'm interested in high-risk, high-reward investment opportunities.",
     "context": "Young, risk-tolerant, and looking for growth. Keen on investing in startups, cryptocurrency, and emerging markets.",
     "question": "What are some of the most promising high-risk, high-reward investments available to young entrepreneurs?"},

    {"about_me": "I'm a 40-year-old single parent. I need to plan for my retirement while supporting my children.",
     "context": "Single income with dual responsibilities. Looking for balanced investment strategies that offer growth and security.",
     "question": "How can I balance my investment strategies to support my children's needs and my own retirement goals?"},

    {"about_me": "I'm a 37-year-old corporate executive. I'm looking to invest in socially responsible companies.",
     "context": "Interested in ethical investing. Focusing on companies with strong ESG (Environmental, Social, Governance) ratings.",
     "question": "What are the best approaches to finding and investing in socially responsible companies?"},

    {"about_me": "I'm a 32-year-old artist. I'm looking for creative ways to invest my modest earnings.",
     "context": "Unconventional income patterns, interested in the arts. Exploring investments in art, collectibles, and niche markets.",
     "question": "What are some unique investment options that align with my artistic background and financial situation?"},

    {"about_me": "I'm a 28-year-old professional gamer. I want to invest my earnings in a way that aligns with my tech-savvy nature.",
     "context": "Tech-oriented with an interest in digital and gaming industries. Looking at tech stocks, e-sports companies, and digital currencies.",
     "question": "How can I leverage my understanding of the gaming and tech industry in my investment choices?"},

    {"about_me": "I'm a 50-year-old teacher nearing retirement. I want to invest in safe, income-generating assets.",
     "context": "Approaching retirement with a conservative mindset. Interested in government bonds, dividend stocks, and annuities.",
     "question": "What are some safe investment options that can provide a steady income during retirement?"},

    {"about_me": "I'm a 39-year-old professional dancer. My career may not last long, and I need to plan for the future.",
     "context": "Potentially short career span with uncertain future earnings. Seeking investments in retirement funds and stable stocks.",
     "question": "What investment strategies would you recommend for someone with a potentially short career lifespan?"},

    {"about_me": "I'm a 27-year-old environmental scientist. I want my investments to reflect my commitment to sustainability.",
     "context": "Passionate about the environment. Looking to invest in renewable energy, sustainable practices, and green technology.",
     "question": "What are some effective ways to invest in the green technology and renewable energy sectors?"},

    {"about_me": "I'm a 43-year-old professional musician. I'm looking for ways to invest my savings from performances and royalties.",
     "context": "Variable income from music career. Interested in a mix of aggressive and conservative investments.",
     "question": "How can I create a balanced investment portfolio that takes into account the irregular nature of my income?"},

    {"about_me": "I'm a 36-year-old real estate agent. I want to expand my investment horizons beyond real estate.",
     "context": "Experienced in real estate, seeking diversification. Interested in the stock market, mutual funds, and ETFs.",
     "question": "What are some effective ways for a real estate professional to diversify investments outside of the real estate market?"},

    {"about_me": "I'm a 31-year-old mechanical engineer. I'm curious about investing in technology and industrial sectors.",
     "context": "Technical background with an interest in industrial innovation. Looking at tech companies, industrial stocks, and sector ETFs.",
     "question": "As an engineer, what sectors or companies should I consider for investment in the technology and industrial fields?"},

    {"about_me": "I'm a 60-year-old retiree looking to manage my investments actively in my retirement years.",
     "context": "Retired with a focus on active investment management. Interested in stocks, bonds, and staying engaged in the market.",
     "question": "What are some strategies for actively managing my investment portfolio during retirement without taking on too much risk?"},
    
    {"about_me": "I'm a 41-year-old civil engineer. I'm looking to invest in infrastructure and construction sectors.",
     "context": "Interest in leveraging professional knowledge in personal investments. Focused on infrastructure and construction industry.",
     "question": "Which aspects of the infrastructure and construction sectors are most promising for investment?"},

    {"about_me": "I'm a 33-year-old stay-at-home parent. I want to invest in a way that allows me to manage family and finances.",
     "context": "Time-flexible but needs low-maintenance investments. Interested in long-term growth and educational funds for children.",
     "question": "What are some suitable investment strategies for a stay-at-home parent focusing on long-term family financial security?"},

    {"about_me": "I'm a 48-year-old airline pilot. I want to secure my financial future given the uncertain nature of my industry.",
     "context": "Concerned about industry stability. Looking for investments that provide security and are less affected by industry volatility.",
     "question": "As an airline pilot, how can I invest to protect myself from potential industry downturns?"},

    {"about_me": "I'm a 26-year-old social media influencer. I want to capitalize on my current earnings for long-term benefits.",
     "context": "High current income with uncertain long-term prospects. Interested in aggressive growth investments and retirement planning.",
     "question": "What are the best investment approaches for someone with an unconventional and potentially fluctuating income like mine?"},

    {"about_me": "I'm a 55-year-old sales manager. I'm looking to diversify my investment portfolio as I approach retirement.",
     "context": "Approaching retirement with a focus on diversification. Interested in a balanced mix of stocks, bonds, and real estate.",
     "question": "How should I adjust my investment portfolio as I get closer to retirement for both growth and safety?"},

    {"about_me": "I'm a 30-year-old environmental activist. I'm keen on investing in sustainable and ethical companies.",
     "context": "Committed to sustainability and ethics in investing. Focused on finding companies with strong environmental records.",
     "question": "What are the key factors to look for when investing in sustainable and ethical companies?"},

    {"about_me": "I'm a 40-year-old academic researcher. I'm looking for stable investments that complement my modest income.",
     "context": "Steady but modest income, seeking stability. Interested in low-risk investments like bonds and dividend-paying stocks.",
     "question": "What investment options are best suited for someone with a stable but modest income who prefers low-risk investments?"},

    {"about_me": "I'm a 37-year-old professional cyclist. I need to plan financially for when I can no longer compete professionally.",
     "context": "Aware of the limited career span. Looking for investments that can provide a stable income post-career.",
     "question": "What are the best investment strategies for a professional athlete with a potentially short career span?"},

    {"about_me": "I'm a 52-year-old retail manager. I want to start seriously planning for my retirement.",
     "context": "Later start in retirement planning. Needs to focus on aggressive saving and smart investments.",
     "question": "What steps should I take to effectively catch up on my retirement planning at this stage of my career?"},

    {"about_me": "I'm a 24-year-old musician. I'm interested in investing a portion of my earnings from performances and royalties.",
     "context": "Irregular income stream. Looking for a mix of growth and stable investments to build wealth over time.",
     "question": "How can I balance my investment portfolio to manage the irregular nature of my income as a musician?"},

    {"about_me": "I'm a 29-year-old startup founder. I want to diversify my investments outside of my business.",
     "context": "High risk associated with business. Looking to diversify into stocks, real estate, and perhaps bonds.",
     "question": "As a startup founder, how can I effectively diversify my personal investments outside of my business venture?"},

    {"about_me": "I'm a 35-year-old pharmacist. I'm interested in healthcare and pharmaceutical investments.",
     "context": "Industry knowledge in healthcare and pharmacy. Interested in investing in health stocks and new medical technologies.",
     "question": "What are some key healthcare and pharmaceutical sectors or companies I should consider for investment?"},

    {"about_me": "I'm a 45-year-old financial analyst. I want to use my skills to build a robust investment portfolio.",
     "context": "Professional financial expertise. Interested in a data-driven approach to investing in stocks and ETFs.",
     "question": "Given my background in finance, how can I leverage my skills to build a successful investment portfolio?"},

    {"about_me": "I'm a 39-year-old high school teacher. I want to start a college fund for my children while securing my retirement.",
     "context": "Balancing saving for children’s education with retirement planning. Interested in 529 plans and retirement accounts.",
     "question": "How can I effectively balance investing in my children's education and my own retirement?"},

    {"about_me": "I'm a 31-year-old graphic designer. I have an interest in digital and media investments.",
     "context": "Creative professional with an interest in the digital media landscape. Exploring investments in digital media and technology.",
     "question": "What are some promising investment opportunities in the digital and media sectors for someone in my field?"},

    {"about_me": "I'm a 53-year-old nurse. I'm looking to invest in health and wellness sectors.",
     "context": "Experience in healthcare. Interested in emerging health and wellness trends, biotech, and healthcare services.",
     "question": "What are the most promising areas within the health and wellness sectors for investment?"},

    {"about_me": "I'm a 28-year-old professional photographer. I want to invest my savings wisely to support my artistic career.",
     "context": "Freelance income, looking for financial stability. Interested in a diversified investment portfolio to support an artistic career.",
     "question": "How can I create a diversified investment strategy that supports the financial demands of my career in photography?"},

    {"about_me": "I'm a 44-year-old event planner. I'm looking for investment options that align with the hospitality and event industry.",
     "context": "Industry knowledge in events and hospitality. Interested in stocks related to tourism, events, and hospitality.",
     "question": "What are some strategic investment options within the hospitality and event sectors?"},

    {"about_me": "I'm a 36-year-old professional soccer player. I want to invest for the future, knowing my career may be short.",
     "context": "High earning but short career span. Focused on securing financial stability post-career through smart investments.",
     "question": "What investment strategies are recommended for a professional athlete with a potentially brief career span?"},

    {"about_me": "I'm a 50-year-old interior designer. I'm interested in real estate and art investments.",
     "context": "Professional interest in aesthetics and design. Exploring investments in real estate, art, and design-related fields.",
     "question": "As an interior designer, what are some unique investment opportunities in art and real estate that I should consider?"},

    {"about_me": "I'm a 32-year-old software engineer. I'm keen on investing in emerging tech and AI companies.",
     "context": "Tech background with an interest in cutting-edge technology. Focused on AI, machine learning, and software startups.",
     "question": "What are some key emerging tech and AI companies that are worth considering for investment?"},

    {"about_me": "I'm a 27-year-old entrepreneur in the fitness industry. I want to invest in health and fitness technology.",
     "context": "Interest in fitness and technology. Looking to invest in wearable tech, fitness apps, and health-related technology startups.",
     "question": "Which areas within health and fitness technology present the best investment opportunities currently?"},

    {"about_me": "I'm a 41-year-old corporate lawyer. I'm interested in legal tech and fintech investments.",
     "context": "Professional experience in law and finance. Exploring investments in legal tech startups and financial technology companies.",
     "question": "What are the emerging opportunities in legal tech and fintech that could be lucrative for investment?"},

    {"about_me": "I'm a 47-year-old commercial pilot. I'm exploring investments related to aviation and aerospace.",
     "context": "Industry insight into aviation and aerospace. Interested in investing in airline stocks, aerospace engineering, and space exploration.",
     "question": "Given my background as a pilot, what are the most promising investment opportunities in the aviation and aerospace sectors?"},

    {"about_me": "I'm a 38-year-old biologist. I'm looking to invest in environmental and green technology sectors.",
     "context": "Background in biology and environmental science. Interested in sustainable investments, green tech, and renewable energy.",
     "question": "As a biologist, what are some of the most impactful investment opportunities in environmental and green technology?"},

    {"about_me": "I'm a 34-year-old marketing professional. I want to invest in digital marketing and e-commerce platforms.",
     "context": "Expertise in digital marketing. Looking at investments in e-commerce, online advertising platforms, and digital marketing tools.",
     "question": "What are some potential investment opportunities in the digital marketing and e-commerce space that align with my profession?"},

      {"about_me": "I'm a 33-year-old civil servant. I'm looking for stable long-term investment options.",
     "context": "Seeking secure investments with steady growth. Interested in government bonds and blue-chip stocks.",
     "question": "What are the best low-risk investment options for long-term stability in my portfolio?"},

    {"about_me": "I'm a 27-year-old professional athlete. My career might be short, and I want to secure my financial future.",
     "context": "High but potentially short-lived earnings. Looking for investments that can provide long-term financial security.",
     "question": "How should I invest my earnings now to ensure financial stability after my sports career ends?"},

    {"about_me": "I'm a 41-year-old single mother. I want to invest for my child's future and my retirement.",
     "context": "Balancing the need to save for a child's education and own retirement. Interested in education savings accounts and retirement funds.",
     "question": "What's the best way to simultaneously invest for my retirement and my child's education?"},

    {"about_me": "I'm a 35-year-old freelance writer. My income varies, so I need flexible investment options.",
     "context": "Irregular income streams. Looking for investments with good growth potential and liquidity.",
     "question": "What are some flexible investment options that suit a freelancer's variable income?"},

    {"about_me": "I'm a 48-year-old corporate executive. I want to diversify my investment portfolio.",
     "context": "High income and looking to diversify investments. Interested in a mix of stocks, real estate, and possibly international investments.",
     "question": "How can I diversify my investment portfolio to reduce risk and ensure long-term growth?"},

    {"about_me": "I'm a 26-year-old software engineer. I'm interested in tech startups and cryptocurrency.",
     "context": "Tech-savvy and willing to take calculated risks. Keen on emerging technologies and digital currencies.",
     "question": "What should I know about investing in tech startups and cryptocurrencies given their volatile nature?"},

    {"about_me": "I'm a 52-year-old dentist. I want to start planning for a secure retirement.",
     "context": "Mid-career professional looking to build retirement savings. Interested in a mix of stocks, bonds, and retirement accounts.",
     "question": "What are the most effective strategies for a mid-career professional to secure a comfortable retirement?"},

    {"about_me": "I'm a 44-year-old marketing director. I'm looking for aggressive growth investment opportunities.",
     "context": "High-income individual seeking high-growth investments. Interested in stock market, private equity, and high-yield bonds.",
     "question": "What are some aggressive investment strategies for someone looking to significantly grow their wealth?"},

    {"about_me": "I'm a 31-year-old nurse. I want to invest in health care and biotechnology sectors.",
     "context": "Industry knowledge in healthcare. Interested in health care stocks, biotech companies, and medical research investments.",
     "question": "Given my background in nursing, what are the most promising investment opportunities in the healthcare and biotech sectors?"},

    {"about_me": "I'm a 39-year-old artist. I'm looking for creative and unconventional investment options.",
     "context": "Interested in aligning investments with artistic values. Exploring investments in art, startups, and niche markets.",
     "question": "What are some unique and creative investment options that align with an artistic career?"},

    {"about_me": "I'm a 37-year-old electrician. I want to invest in the energy sector, particularly in renewables.",
     "context": "Interest in the energy sector, especially renewable energy. Looking to invest in solar, wind, and other green technologies.",
     "question": "What are the best investment strategies in the renewable energy sector for someone with my background?"},

    {"about_me": "I'm a 29-year-old HR manager. I'm interested in ethical and socially responsible investing.",
     "context": "Focused on ESG (Environmental, Social, Governance) investing. Looking for companies with strong ethical values.",
     "question": "How can I find and invest in companies that meet high ESG standards?"},

    {"about_me": "I'm a 55-year-old professor. I'm looking for ways to supplement my income as I near retirement.",
     "context": "Approaching retirement with a focus on additional income sources. Interested in dividend stocks and real estate investments.",
     "question": "What are some effective investment strategies for supplementing income in retirement?"},

    {"about_me": "I'm a 46-year-old airline pilot. I want to invest in travel and aviation sectors.",
     "context": "Industry knowledge in aviation and travel. Interested in airline stocks, airport services, and travel technology companies.",
     "question": "Given my experience as a pilot, what are the best investment opportunities in the aviation and travel sectors?"},

    {"about_me": "I'm a 50-year-old entrepreneur. I want to invest in emerging markets and international businesses.",
     "context": "Looking to expand investment horizons globally. Interested in emerging markets, international stocks, and foreign real estate.",
     "question": "What should I consider when investing in emerging markets and international businesses?"},

    {"about_me": "I'm a 28-year-old environmental consultant. I'm looking to invest in sustainable and eco-friendly businesses.",
     "context": "Passionate about sustainability. Interested in companies that focus on environmental solutions and sustainable practices.",
     "question": "How can I identify and invest in companies that are genuinely committed to sustainability and eco-friendliness?"},

    {"about_me": "I'm a 43-year-old professional musician. I'm interested in income-generating investments to support my artistic career.",
     "context": "Looking for investments that provide a steady income. Interested in dividend stocks, bonds, and real estate.",
     "question": "What are the best investment options for generating a steady income to support an artistic career?"},

    {"about_me": "I'm a 38-year-old sports coach. I want to start investing in health and fitness-related industries.",
     "context": "Interested in the health and fitness industry. Looking at sports companies, fitness technology, and wellness startups.",
     "question": "What are the most promising investment opportunities in the health and fitness industries for someone with my background?"},

    {"about_me": "I'm a 60-year-old retiree. I want to invest conservatively but still see some growth.",
     "context": "Retired and focused on preserving capital while earning some returns. Interested in low-risk investments with modest growth potential.",
     "question": "What conservative investment strategies can still offer some growth for a retiree like me?"},

    {"about_me": "I'm a 34-year-old project manager. I want to start investing in tech and innovation sectors.",
     "context": "Keen interest in technology and innovation. Looking to invest in tech startups, AI companies, and innovative enterprises.",
     "question": "As a project manager with an interest in tech, what are the best areas in innovation to invest in?"},

    {"about_me": "I'm a 22-year-old recent college graduate. I want to start investing early in my career.",
     "context": "Beginning of career with long-term investment horizon. Interested in stocks, mutual funds, and retirement accounts.",
     "question": "What are the best investment strategies for someone just starting their career and looking to invest long-term?"},

    {"about_me": "I'm a 49-year-old architect. I'm looking for mid-risk investment opportunities.",
     "context": "Mid-career professional seeking a balance between risk and security. Interested in a diversified portfolio with both stocks and bonds.",
     "question": "What are some balanced mid-risk investment strategies suitable for someone in my profession and age group?"},

    {"about_me": "I'm a 26-year-old social media influencer. I'm interested in investing in media and entertainment sectors.",
     "context": "Experience in digital media. Looking to invest in media companies, entertainment startups, and digital marketing firms.",
     "question": "How can I leverage my background in social media to invest effectively in the media and entertainment sectors?"},

    {"about_me": "I'm a 53-year-old lawyer. I want to create a diversified investment portfolio.",
     "context": "Established career with a focus on wealth accumulation. Interested in a mix of stocks, bonds, real estate, and possibly gold.",
     "question": "What is the best way to create a diversified and balanced investment portfolio at this stage of my career?"},

    {"about_me": "I'm a 40-year-old chef. I want to invest in the food and beverage industry.",
     "context": "Industry knowledge in food and beverage. Interested in restaurant stocks, food technology, and sustainable food startups.",
     "question": "What are the best investment opportunities in the food and beverage industry for a professional chef like me?"},

    {"about_me": "I'm a 31-year-old graphic designer. I want to start investing in creative industries.",
     "context": "Interest in creative and design industries. Looking at advertising firms, design startups, and digital media companies.",
     "question": "As a graphic designer, what are the most promising investment opportunities in the creative industries?"},
         {"about_me": "I'm a 33-year-old event planner. I want to start investing for my future and possibly start my own business.",
     "context": "Interested in building savings for future business endeavors. Looking for investments that offer both growth potential and flexibility.",
     "question": "What investment options should I consider that would support my goal of starting a business in the future?"},

    {"about_me": "I'm a 41-year-old pilot. I have a stable job and want to ensure financial security for my family.",
     "context": "Seeking long-term financial security with a focus on children’s education and retirement. Interested in a balanced mix of investments.",
     "question": "How can I create a diversified investment plan that covers both my children's education and my retirement?"},

    {"about_me": "I'm a 29-year-old biologist. I want to invest in a way that aligns with my passion for environmental conservation.",
     "context": "Passionate about the environment, looking for green investments. Interested in sustainable funds and clean energy stocks.",
     "question": "What are the best investment options for someone who wants to focus on environmental sustainability?"},

    {"about_me": "I'm a 48-year-old sales manager. I'm looking for investment strategies to maximize my earnings before retirement.",
     "context": "Approaching retirement age and looking to maximize savings. Interested in a mix of stocks, bonds, and perhaps some higher-risk investments.",
     "question": "What are some effective investment strategies for someone in their late 40s to maximize retirement savings?"},

    {"about_me": "I'm a 35-year-old digital nomad, working in various countries. I'm looking for flexible and international investment options.",
     "context": "Frequent traveler seeking investment opportunities that are not geographically limited. Interested in global stocks and digital assets.",
     "question": "What are some global investment options that would suit a digital nomad lifestyle?"},

    {"about_me": "I'm a 26-year-old social media influencer. I want to invest my earnings wisely in this volatile industry.",
     "context": "Earning well in a rapidly changing industry. Looking for investments that can offer stability and long-term growth.",
     "question": "What investment strategies would be best for someone with an unpredictable income in a volatile industry?"},

    {"about_me": "I'm a 52-year-old doctor. I want to diversify my investments to include more than just my medical practice.",
     "context": "Established career but seeking investment diversification. Interested in stocks, real estate, and possibly venture capital.",
     "question": "How can I diversify my investment portfolio beyond my medical practice?"},

    {"about_me": "I'm a 44-year-old professional painter. I want to invest in a way that might also benefit my artistic career.",
     "context": "Artist interested in investments related to the art world. Considering art funds, gallery investments, and creative startups.",
     "question": "What are some creative investment options that could also potentially benefit my career as an artist?"},

    {"about_me": "I'm a 39-year-old firefighter. I'm looking for secure, long-term investment options for my retirement.",
     "context": "Seeking secure investments for post-retirement. Interested in government bonds, pension funds, and real estate.",
     "question": "What are some of the safest long-term investment options for a firefighter approaching retirement?"},

    {"about_me": "I'm a 31-year-old environmental lobbyist. I want my investments to reflect my values for a sustainable future.",
     "context": "Passionate about sustainability and ethical investing. Looking to invest in ESG funds and sustainable companies.",
     "question": "How can I align my investment strategy with my commitment to environmental sustainability?"},

    {"about_me": "I'm a 37-year-old stay-at-home parent. I want to start investing to contribute to our family's financial health.",
     "context": "Looking to invest for family's future, including education and retirement. Interested in long-term, stable investments.",
     "question": "What are some wise investment choices for a stay-at-home parent focusing on long-term family financial health?"},

    {"about_me": "I'm a 28-year-old professional cyclist. I want to invest my prize earnings for future financial security.",
     "context": "Young athlete with potentially short career span. Looking for ways to secure financial stability post-athletic career.",
     "question": "How should I invest my earnings now to ensure financial security after my athletic career ends?"},

    {"about_me": "I'm a 46-year-old journalist specializing in international affairs. I'm looking for global investment opportunities.",
     "context": "Interested in a diverse, global investment portfolio. Looking at emerging markets, international funds, and foreign currencies.",
     "question": "What are some promising global investment opportunities for someone with international expertise?"},

    {"about_me": "I'm a 50-year-old car dealership owner. I want to invest profits from my business in a smart way.",
     "context": "Business owner seeking to reinvest profits. Interested in stock market, real estate, and potentially high-risk ventures.",
     "question": "How can I best invest the profits from my car dealership to ensure continued financial growth?"},

    {"about_me": "I'm a 33-year-old software engineer with a passion for AI and machine learning. I want to invest in the tech industry.",
     "context": "Tech-savvy individual interested in cutting-edge technology investments. Focusing on AI, machine learning, and tech startups.",
     "question": "What are some strategic investment options in the AI and machine learning sectors?"},

    {"about_me": "I'm a 40-year-old professional golfer. I want to start planning for financial stability beyond my sports career.",
     "context": "Sports professional aware of the limited career span. Interested in stable investments and retirement planning.",
     "question": "What are the best investment strategies for a professional athlete to secure financial stability post-career?"},

    {"about_me": "I'm a 27-year-old fashion designer. I'm interested in investing in the fashion and luxury goods market.",
     "context": "Fashion industry professional looking to invest in related markets. Interested in luxury brands, retail stocks, and fashion startups.",
     "question": "What are some savvy investment options within the fashion and luxury goods market?"},

    {"about_me": "I'm a 55-year-old accountant. I want to use my financial skills to build a strong investment portfolio.",
     "context": "Financially knowledgeable and looking for sophisticated investment strategies. Considering stocks, bonds, and tax-efficient investing.",
     "question": "What are some advanced investment strategies that would be suitable for a finance professional?"},

    {"about_me": "I'm a 35-year-old professional skateboarder. I want to invest in industries that align with my lifestyle and interests.",
     "context": "Interested in lifestyle-aligned investments, such as sports equipment companies, lifestyle brands, and youth culture markets.",
     "question": "What are some investment opportunities that align with the skateboarding and youth culture lifestyle?"},

    {"about_me": "I'm a 42-year-old film director. I want to explore investment opportunities in the entertainment industry.",
     "context": "Industry professional interested in entertainment sector investments. Considering film production companies, streaming services, and media stocks.",
     "question": "As a film director, what are some strategic investments I can make in the entertainment industry?"},

    {"about_me": "I'm a 30-year-old marine biologist. I'm interested in investments that contribute to ocean conservation.",
     "context": "Passionate about marine life and ocean health. Looking to invest in sustainable fisheries, ocean cleanup companies, and environmental funds.",
     "question": "How can I invest in companies or funds that are actively contributing to ocean conservation?"},

    {"about_me": "I'm a 24-year-old esports competitor. I want to invest my winnings in a way that secures my financial future.",
     "context": "Young and successful in a rapidly growing industry. Interested in tech stocks, cryptocurrency, and gaming industry investments.",
     "question": "What are the most effective investment strategies for someone in the burgeoning esports industry?"},

    {"about_me": "I'm a 39-year-old veterinarian. I'm looking to invest in the animal health and veterinary sectors.",
     "context": "Professional experience in animal healthcare. Interested in investing in veterinary companies, pet care startups, and related industries.",
     "question": "What are some smart investment choices in the animal health and veterinary sectors?"},

    {"about_me": "I'm a 48-year-old mining engineer. I'm interested in investing in natural resources and sustainable mining practices.",
     "context": "Industry knowledge in mining and natural resources. Looking at mineral stocks, sustainable resource companies, and green mining technology.",
     "question": "What are the best investment strategies for someone with expertise in the mining and natural resources sector?"},

    {"about_me": "I'm a 26-year-old professional blogger. I want to invest in digital media and online platforms.",
     "context": "Digital savvy individual interested in the online media landscape. Focusing on digital media companies, online advertising, and content platforms.",
     "question": "As a professional blogger, what are some promising investment opportunities in digital media and online platforms?"},
         {"about_me": "I'm a 44-year-old commercial pilot. I've been focusing on saving for retirement and looking into diverse investment options.",
     "context": "Steady income with a focus on retirement planning. Interested in a mix of traditional and alternative investments.",
     "question": "What are some unique investment strategies suitable for someone in the aviation industry?"},

    {"about_me": "I'm a 29-year-old social media influencer. My income is heavily tied to the digital economy.",
     "context": "Earning through digital platforms, interested in investing in the digital and social media market.",
     "question": "How can I invest wisely in the digital market, given my profession as a social media influencer?"},

    {"about_me": "I'm a 39-year-old veterinarian. I'm looking for investment options that align with my love for animals and nature.",
     "context": "Passionate about animal welfare and environmental conservation. Exploring ethical and sustainable investment opportunities.",
     "question": "What are some animal and environment-friendly investment options available for someone in my field?"},

    {"about_me": "I'm a 31-year-old electrician. I want to invest in a way that can secure a stable financial future.",
     "context": "Skilled trade with a steady income. Interested in straightforward, low-risk investment strategies.",
     "question": "As a tradesperson, what are some simple yet effective investment strategies I can adopt?"},

    {"about_me": "I'm a 50-year-old film director. I want to invest in creative industries and media.",
     "context": "Industry experience in film and media. Looking to invest in entertainment, media, and possibly tech startups.",
     "question": "How can I strategically invest in the media and entertainment industries?"},

    {"about_me": "I'm a 27-year-old professional athlete. I'm interested in investing in sports-related businesses.",
     "context": "Earnings from sports, with a keen interest in sports business ventures. Considering investments in sports tech and merchandise.",
     "question": "What are some promising investment opportunities within the sports industry?"},

    {"about_me": "I'm a 48-year-old academic researcher. I'm looking to invest in education and research sectors.",
     "context": "Background in academia. Interested in investing in educational technology and research companies.",
     "question": "How can I align my investment strategy with my professional background in academia?"},

    {"about_me": "I'm a 35-year-old logistics manager. I want to invest in the growing field of logistics and supply chain.",
     "context": "Experience in logistics and supply chain management. Looking at companies in logistics, e-commerce, and related tech.",
     "question": "What are the best investment opportunities in the logistics and supply chain industry?"},

    {"about_me": "I'm a 60-year-old retired military officer. I'm interested in investments that are secure and offer steady income.",
     "context": "Retired with a pension, seeking additional income sources. Focused on low-risk, stable investments.",
     "question": "What are some secure investment strategies suitable for a retired military officer?"},

    {"about_me": "I'm a 41-year-old graphic designer. I'm curious about investing in technology and the creative industry.",
     "context": "Background in design and tech. Interested in investing in tech startups, digital media, and creative ventures.",
     "question": "As a graphic designer, what are some unique areas in tech and creativity where I can invest?"},

    {"about_me": "I'm a 26-year-old biologist. I want to start investing in environmental and sustainable projects.",
     "context": "Passionate about environmental science. Interested in sustainable energy, green tech, and conservation projects.",
     "question": "How can I find and invest in sustainable and environmentally-friendly projects?"},

    {"about_me": "I'm a 55-year-old dentist. I'm looking for ways to invest in the healthcare sector.",
     "context": "Professional experience in healthcare. Interested in healthcare technology, biotech startups, and pharmaceuticals.",
     "question": "What are some investment opportunities in the healthcare sector that a professional like me should consider?"},

    {"about_me": "I'm a 33-year-old fashion designer. I'm interested in investing in the fashion and luxury goods market.",
     "context": "Industry knowledge in fashion and luxury. Looking to invest in high-end fashion brands, retail, and related tech.",
     "question": "What are some savvy investment strategies in the fashion and luxury goods sector?"},

    {"about_me": "I'm a 47-year-old oil and gas industry worker. I'm looking to diversify my investments outside of my industry.",
     "context": "Experience in the energy sector but seeking diversification. Interested in renewable energy and technology.",
     "question": "As someone from the oil and gas industry, how can I diversify my investments effectively?"},

    {"about_me": "I'm a 30-year-old event planner. I want to invest in ways that could complement my career.",
     "context": "Career in event management. Looking at investments in hospitality, entertainment, and real estate.",
     "question": "What are some strategic investment options that align with my career in event planning?"},

    {"about_me": "I'm a 52-year-old journalist. I'm interested in media and information technology investments.",
     "context": "Years of experience in media. Interested in digital media companies, online platforms, and information technology.",
     "question": "What are some wise investment choices in the media and information technology sectors?"},

    {"about_me": "I'm a 38-year-old environmental consultant. I want to invest in eco-friendly and sustainable businesses.",
     "context": "Expertise in environmental consulting. Focused on investments in eco-friendly businesses and sustainable practices.",
     "question": "What are the best options for investing in eco-friendly and sustainable businesses?"},

    {"about_me": "I'm a 28-year-old professional musician. I'm looking for investment opportunities in the music industry.",
     "context": "Earning from music, interested in music industry investments. Looking at music streaming services, labels, and tech.",
     "question": "How can I invest in the music industry in a way that complements my career as a musician?"},

    {"about_me": "I'm a 45-year-old HR professional. I'm looking for stable and long-term investment options.",
     "context": "Steady career in human resources. Interested in long-term, low-risk investments for retirement.",
     "question": "What are some low-risk, long-term investment strategies suitable for an HR professional?"},

    {"about_me": "I'm a 37-year-old software engineer. I'm interested in tech startups and the cryptocurrency market.",
     "context": "Tech-savvy with a high-risk appetite. Looking at early-stage tech startups and cryptocurrency investments.",
     "question": "What are the key factors I should consider when investing in tech startups and cryptocurrencies?"},

    {"about_me": "I'm a 49-year-old professional chef. I'm interested in the food and beverage industry investments.",
     "context": "Experience in the culinary industry. Looking to invest in food tech, organic food companies, and restaurant chains.",
     "question": "How can I smartly invest in the food and beverage industry, given my background as a chef?"},

    {"about_me": "I'm a 34-year-old car mechanic. I'm interested in investing in the automotive and transportation sector.",
     "context": "Background in automotive repair and maintenance. Interested in automotive stocks, electric vehicles, and related tech.",
     "question": "What are some promising investment opportunities in the automotive and transportation sector?"},

    {"about_me": "I'm a 42-year-old real estate developer. I'm looking for investment opportunities beyond real estate.",
     "context": "Expertise in real estate development but seeking diversification. Interested in stock market and alternative investments.",
     "question": "How can a real estate developer like me diversify investments beyond the property market?"},

    {"about_me": "I'm a 36-year-old marketing director. I'm looking to invest in digital marketing and e-commerce businesses.",
     "context": "Professional experience in marketing. Focused on investments in digital marketing startups and e-commerce.",
     "question": "What are the best investment strategies for someone with expertise in digital marketing and e-commerce?"},

    {"about_me": "I'm a 50-year-old police officer. I want to start planning for my retirement with secure investments.",
     "context": "Seeking financial security post-retirement. Interested in low-risk investments like government bonds and dividend stocks.",
     "question": "What are some safe and secure investment options for a police officer nearing retirement?"},

    {"about_me": "I'm a 26-year-old professional gamer. I'm interested in investing in the gaming and tech industry.",
     "context": "Earnings from gaming, with a passion for technology. Looking to invest in gaming companies, e-sports, and tech startups.",
     "question": "What are some strategic investment opportunities in the gaming and technology industry?"},

    {"about_me": "I'm a 41-year-old firefighter. I'm looking for ways to invest my savings for a stable financial future.",
     "context": "Steady income as a firefighter. Interested in risk-averse investment strategies for long-term stability.",
     "question": "What are some low-risk investment options that would be appropriate for a firefighter?"},
       {"about_me": "I'm a 33-year-old environmental activist. I want my investments to reflect my values.",
     "context": "Seeking investments in sustainable and eco-friendly companies. Interested in avoiding fossil fuels and supporting green technology.",
     "question": "What are some effective investment strategies for someone committed to environmental activism?"},

    {"about_me": "I'm a 40-year-old veterinarian. I'm interested in animal welfare and would like my investments to align with this passion.",
     "context": "Looking for ethical investments, particularly in companies that promote animal welfare or veterinary science.",
     "question": "Can you suggest investment opportunities that support animal welfare and are financially sound?"},

    {"about_me": "I'm a 28-year-old professional cyclist. I want to invest my prize money wisely for the future.",
     "context": "Seeking to invest in a way that considers the potentially short duration of a sports career. Interested in stable, long-term options.",
     "question": "What are some prudent investment choices for a professional athlete with an unpredictable career lifespan?"},

    {"about_me": "I'm a 50-year-old corporate trainer. I've been focused on my career and need to catch up on retirement planning.",
     "context": "Mid-life stage with a moderate retirement fund. Looking for aggressive yet safe investment strategies to grow retirement savings.",
     "question": "How can I accelerate my retirement savings at this stage of my life?"},

    {"about_me": "I'm a 35-year-old social worker. I have a modest income and want to start investing for the future.",
     "context": "Steady but modest income, seeking simple and effective investment strategies. Interested in a mix of stocks and bonds.",
     "question": "What are some low-risk investment options for someone with a modest income to start building wealth?"},

    {"about_me": "I'm a 26-year-old startup founder. I want to diversify my investments outside of my business.",
     "context": "High-risk, high-reward business venture. Looking to balance this with more stable and diversified personal investments.",
     "question": "As a startup founder, how can I diversify my personal investment portfolio to mitigate risk?"},

    {"about_me": "I'm a 42-year-old academic researcher. I'm looking for ways to invest in education and research sectors.",
     "context": "Interested in supporting and investing in education and research-focused companies. Seeking both financial returns and to support these sectors.",
     "question": "What investment opportunities are available in the education and research sectors?"},

    {"about_me": "I'm a 38-year-old professional chef. I'm interested in investing in the food and hospitality industry.",
     "context": "Industry expertise in food and hospitality. Looking for investment opportunities in these sectors, possibly including startups.",
     "question": "What are some promising investment opportunities in the food and hospitality industry?"},

    {"about_me": "I'm a 31-year-old digital nomad. I travel frequently and work remotely, and I'm looking for flexible investment options.",
     "context": "Non-traditional career with a global perspective. Interested in digital assets and international investments.",
     "question": "What are some suitable investment options for someone with a digital nomad lifestyle?"},

    {"about_me": "I'm a 48-year-old police officer. I want to make wise investments for my post-retirement life.",
     "context": "Approaching retirement with a pension. Looking for additional income sources and investments to secure financial stability post-retirement.",
     "question": "What are some safe and reliable investment strategies for a soon-to-be-retired police officer?"},

    {"about_me": "I'm a 29-year-old event planner. I want to start investing in a way that could also help expand my business.",
     "context": "Looking to grow personal wealth and business simultaneously. Interested in investments that could benefit the event planning industry.",
     "question": "How can I align my investment strategy to support both my personal financial growth and my event planning business?"},

    {"about_me": "I'm a 54-year-old dentist. I'm looking for conservative investment strategies as I approach retirement.",
     "context": "Close to retirement, seeking to preserve wealth. Focused on low-risk investments that offer stability and consistent returns.",
     "question": "What are the best conservative investment strategies for a professional nearing retirement?"},

    {"about_me": "I'm a 37-year-old fashion designer. I want to invest in the fashion industry and related sectors.",
     "context": "Industry knowledge in fashion. Looking to invest in fashion startups, retail companies, and sustainable fashion technologies.",
     "question": "What are some smart investment opportunities within the fashion industry and its emerging trends?"},

    {"about_me": "I'm a 30-year-old journalist. I'm interested in media and communications companies.",
     "context": "Industry experience in media and journalism. Looking to invest in media companies, especially those embracing new technologies.",
     "question": "What are some key considerations when investing in media and communications companies?"},

    {"about_me": "I'm a 40-year-old psychologist. I want to invest in health and wellness sectors.",
     "context": "Professional experience in mental health. Interested in companies that focus on wellness, mental health technology, and healthcare services.",
     "question": "What investment opportunities exist in the health and wellness sectors that align with my professional expertise?"},

    {"about_me": "I'm a 25-year-old professional gamer. I want to invest in the gaming industry and esports.",
     "context": "Deep understanding of the gaming industry. Interested in investing in gaming companies, esports leagues, and related technology.",
     "question": "What are some promising investment avenues within the gaming and esports industry?"},

    {"about_me": "I'm a 35-year-old environmental engineer. I'm interested in green technology and sustainable energy investments.",
     "context": "Passionate about environmental sustainability. Looking to invest in renewable energy, green technology startups, and sustainable initiatives.",
     "question": "What are the best ways to invest in green technology and sustainable energy considering my background?"},

    {"about_me": "I'm a 32-year-old airline pilot. I want to make smart investments for my future.",
     "context": "Steady income with a focus on long-term financial security. Interested in a diversified portfolio with a mix of stocks, bonds, and real estate.",
     "question": "What are some balanced investment strategies for an airline pilot with long-term financial goals?"},

    {"about_me": "I'm a 27-year-old professional singer. I want to invest my earnings in a way that secures my future.",
     "context": "Irregular income with potential for high earnings. Looking for a combination of aggressive and conservative investment strategies.",
     "question": "How can I balance high-risk and safe investments to ensure financial security as a professional artist?"},

    {"about_me": "I'm a 50-year-old film director. I want to invest in the entertainment industry and beyond.",
     "context": "Industry insight into entertainment and media. Interested in film production companies and emerging digital media platforms.",
     "question": "What are some strategic investment opportunities in the entertainment industry and related sectors?"},

    {"about_me": "I'm a 44-year-old civil engineer. I want to invest in infrastructure and construction sectors.",
     "context": "Professional experience in civil engineering. Looking to invest in infrastructure projects, construction companies, and real estate development.",
     "question": "What are some key investment areas in the infrastructure and construction sectors suitable for someone in my field?"},

    {"about_me": "I'm a 39-year-old HR professional. I'm looking to invest in a way that supports my career growth.",
     "context": "Understanding of corporate environments. Interested in investing in HR technology companies and corporate training providers.",
     "question": "What are some investment options that align with my HR career and have growth potential?"},

    {"about_me": "I'm a 28-year-old marine biologist. I want to invest in ocean conservation and related technologies.",
     "context": "Passionate about marine life and ocean health. Looking to invest in sustainable fisheries, ocean technology, and conservation projects.",
     "question": "What are some investment opportunities that focus on ocean conservation and are financially viable?"},

    {"about_me": "I'm a 31-year-old graphic designer. I want to invest in the technology and design sectors.",
     "context": "Keen interest in technology and design innovation. Looking to invest in tech companies, design startups, and creative industry tools.",
     "question": "What are some promising investment opportunities in the technology and design fields?"},

    {"about_me": "I'm a 46-year-old pharmacist. I'm interested in pharmaceutical and healthcare investments.",
     "context": "Professional experience in pharmacy. Looking to invest in pharmaceutical companies, biotech startups, and healthcare technology.",
     "question": "What are some strategic investment opportunities in the pharmaceutical and healthcare sectors?"},

    {"about_me": "I'm a 53-year-old professor. I want to invest in education technology and academic research companies.",
     "context": "Academic background with an interest in educational advancements. Looking to invest in edtech startups and research-driven companies.",
     "question": "What are some key investment opportunities in education technology and academic research sectors?"},

]


OpenAI.api_key = os.environ["OPENAI_API_KEY"]

def build_prompt(example: Dict) -> str:
    return PROMPT_TEMPLATE.format(
        ABOUT_ME = example["about_me"],
        CONTEXT = example["context"], 
        QUESTION = example["question"], 
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

    


        