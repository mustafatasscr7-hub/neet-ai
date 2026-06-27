import fitz

for pdf, pages in [('AIPMT_2013.pdf', [0,1,2]), ('QP_2023.pdf', [0,1,2])]:
    print(f"\n{'='*60}")
    print(f"PDF: {pdf}")
    print('='*60)
    doc = fitz.open(rf"C:\Users\hakim\Documents\neet-ai\pyq-pdfs\{pdf}")
    for i in pages:
        print(f"\n--- PAGE {i+1} ---")
        print(doc[i].get_text())