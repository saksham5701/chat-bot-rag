from pypdf import PdfReader

pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)
total_pages = len(reader.pages)
print(f"Total pages in the PDF: {total_pages}")

first_page_text = reader.pages[0].extract_text() or ""
print("\nText from the first page:\n")
print(first_page_text[:1250])
print(f"\nPage 1 Characters: {len(first_page_text)}")

total_characters = 0

for page in reader.pages:
    page_text = page.extract_text() or ""
    total_characters += len(page_text)

print(f"\nTotal characters in the PDF: {total_characters}")