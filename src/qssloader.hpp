#ifndef QSSLOADER_HPP
#define QSSLOADER_HPP
#include <QFile>

class QssLoader
{
public:
    static QString load(const QString &filePath)
    {
        QString style;
        QFile file(filePath);
        if (file.open(QFile::ReadOnly)) {
            style = file.readAll();
            file.close();
        }
        return style;
    }
};

#endif // QSSLOADER_HPP
