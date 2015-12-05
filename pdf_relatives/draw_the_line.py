#coding=utf8
from pyPdf import PdfFileWriter, PdfFileReader

pdf = PdfFileReader(file('/home/shin/Memect/ntdb/一方电气股份有限公司公开转让说明书_50.pdf', 'rb'))
out = PdfFileWriter()

for page in pdf.pages:
    page.mediaBox.upperRight = (580,800)
    page.mediaBox.lowerLeft = (128,232)

    out.addPage(page)

cc=pdf.pages[2]
watermark = PdfFileReader(file("/home/shin/Memect/ntdb/ntdb/pdfanalyzer/report.pdf", "rb"))
cc.mergePage(watermark.pages[0])
out.addPage(cc)

ous = file('/home/shin/Memect/ntdb/target.pdf', 'wb')
out.write(ous)
ous.close()